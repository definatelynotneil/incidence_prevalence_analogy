import sys
import os
import ctypes
import datetime
import csv
import gc
import multiprocessing as mp
from itertools import repeat
import logging
from re import sub
import yaml
import resource

import polars as pl
import pyarrow.dataset as ds
from main.preprocessing_functions import process_imd, rmDup, mergeCols, combineLevels, link_hes, create_batch_parquet_files, derive_columns, coalesce_bd_source_cols

def _mem_gb() -> str:
    """Return current RSS memory usage as a formatted string.

    Reads VmRSS from /proc/self/status (current RSS) rather than
    resource.getrusage().ru_maxrss, which on Linux is the lifetime *peak*
    and never decreases — making it useless for confirming memory was freed.
    """
    try:
        with open("/proc/self/status") as _f:
            for _line in _f:
                if _line.startswith("VmRSS:"):
                    return f"{int(_line.split()[1]) / 1024 / 1024:.1f} GB"
    except Exception:
        pass
    # Fallback (macOS or /proc unavailable)
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss_bytes / (1024 ** 2) if sys.platform != "darwin" else rss_bytes / (1024 ** 3)
    return f"{rss_gb:.1f} GB"


def _checkpoint(label: str, logger) -> None:
    msg = f"[MEM {_mem_gb()}] {label}"
    print(msg, flush=True)
    logger.info(msg)


def _free_memory() -> None:
    """Release Python objects and return freed pages to the OS.

    gc.collect() drops Python references. malloc_trim(0) tells the allocator
    to return freed arena pages to the OS. We try CDLL(None) first so that
    if Polars' Rust runtime has loaded jemalloc into the process namespace,
    we call its malloc_trim (which purges dirty pages) rather than glibc's
    (which has no effect on jemalloc arenas).
    """
    gc.collect()
    for _lib in (None, "libc.so.6"):
        try:
            ctypes.CDLL(_lib).malloc_trim(0)
        except Exception:
            pass


def _formNulls_csv_to_parquet(
    in_path: str,
    out_path: str,
    include_columns: list,
    rename_map: dict,
    logger,
    block_mb: int = 128,
) -> None:
    """Stream a wide CSV → parquet using PyArrow's block-wise CSV reader.

    PyArrow's C++ reader skips parsing columns absent from include_columns,
    so peak memory is proportional to the selected columns only — not the
    full width of the CSV. Polars' scan_csv streaming can silently fall back
    to a full collect for certain query plans, causing the entire file to be
    materialised; this function avoids that path entirely.

    block_mb: raw CSV bytes read per batch. Peak RSS per batch ≈
    block_mb × (n_selected / n_total) × parse overhead.
    """
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    import pyarrow.parquet as pq

    final_names = [rename_map.get(c, c) for c in include_columns]
    schema = pa.schema([(name, pa.string()) for name in final_names])

    convert_opts = pa_csv.ConvertOptions(
        include_columns=include_columns,
        null_values=[""],
        strings_can_be_null=True,
        column_types={c: pa.string() for c in include_columns},
    )
    read_opts = pa_csv.ReadOptions(block_size=block_mb * 1024 * 1024)

    reader = pa_csv.open_csv(in_path, read_options=read_opts, convert_options=convert_opts)
    with pq.ParquetWriter(out_path, schema) as writer:
        for batch in reader:
            writer.write_batch(batch.rename_columns(final_names))

    logger.info(f"  _formNulls_csv_to_parquet: wrote {out_path} ({len(include_columns)} cols)")


def _norm_bd_frag(col: str) -> str:
    """Normalise a BD_ column name or fragment for cross-source comparison.

    Strips BD_MEDI: prefix, CPRDAURUM_ and CPRD_ source sub-prefixes,
    removes underscores, and uppercases so that Gold fragments
    (e.g. "CPRD_ACTINIC_KERATOSIS") and Aurum-format column names
    (e.g. "BD_MEDI:CPRDAURUM_ACTINICKERATOSIS") compare equal.
    """
    body = sub(r"^BD_MEDI:CPRDAURUM_", "", col)
    body = sub(r"^BD_MEDI:CPRD_", "", body)
    body = sub(r"^BD_MEDI:", "", body)
    body = sub(r"^CPRDAURUM_", "", body)
    body = sub(r"^CPRD_", "", body)
    return body.replace("_", "").upper()


def _load_condition_map(map_path: str) -> dict:
    """Load condition mapping CSV.

    Expected columns: 'Paper Short Name', 'Gold', 'Aurum'.
    'Gold' and 'Aurum' are column-name fragments (without the 'BD_MEDI:' prefix
    and without the Dexter ':N' numeric suffix).  Empty strings mean the
    condition does not exist in that source.

    Returns {paper_short_name: {'gold': gold_frag, 'aurum': aurum_frag}}.
    """
    result = {}
    with open(map_path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["Paper Short Name"].strip()
            result[name] = {
                "gold": sub(r":\d+$", "", row.get("Gold", "").strip()),
                "aurum": sub(r":\d+$", "", row.get("Aurum", "").strip()),
            }
    return result


def preprocessing(
        dir_data: str,
        config_preproc: dict,
        date_fmt: str = "%Y-%m-%d",
        path_log: str = "log_sBatch_1Python.txt",
        config_incprev: dict = None,
        ) -> None:
    ## Log
    logging.basicConfig(filename=path_log,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger()
    ##

    # Limit Polars streaming chunk size so each sink_parquet pass buffers at
    # most ~50 K rows at a time. The default is large enough that a 600-column
    # CSV can peak at 37 GB even when only 30 columns are selected. This env
    # var is read by Polars when it builds each streaming plan, not at import
    # time, so setting it here takes effect for all subsequent sink calls.
    os.environ["POLARS_STREAMING_CHUNK_SIZE"] = "50000"

    flag_temp_file: bool = False

    _checkpoint("preprocessing() started", logger)

    ## Format Null ################################################################
    logger.info("Formatting null values")

    # Build a set of BD_ column names actually needed (from BD_LIST in incprev config).
    # When the raw CSV has hundreds of condition columns but only a few are needed,
    # reading only those columns avoids loading the entire wide file into RAM.
    #
    # When condition_map_file is set, BD_LIST entries are Paper Short Names looked
    # up in the mapping CSV.  The mapping gives exact Gold/Aurum column fragments;
    # we prepend 'BD_MEDI:' and match exactly (after stripping Dexter ':N' suffix).
    # When no mapping file is given, BD_LIST entries are treated as direct column
    # names and the legacy fuzzy fallback is used (backward-compatible).
    bd_filter: set | None = None
    _use_exact_filter: bool = False
    output_names: dict | None = None   # {norm_key → Paper Short Name} for coalesce step

    if config_incprev and config_incprev.get("BD_LIST"):
        bd_list_entries = [str(b) for b in config_incprev["BD_LIST"]]

        cmap_file = config_preproc.get("condition_map_file")
        if cmap_file:
            cmap = _load_condition_map(f"{dir_data}{cmap_file}")
            col_filter_set: set = set()
            output_names = {}
            for name in bd_list_entries:
                if name not in cmap:
                    logger.warning(
                        f"BD_LIST entry '{name}' not found in condition_map_file; skipping"
                    )
                    continue
                frags = cmap[name]
                gold_frag = frags["gold"]
                aurum_frag = frags["aurum"]
                if gold_frag:
                    col_filter_set.add(f"BD_MEDI:{gold_frag}")
                    # norm_key derived from Gold fragment to match coalesce_bd_source_cols
                    _body = sub(r"^CPRD_", "", gold_frag)
                    output_names[_body.replace("_", "").upper()] = name
                elif aurum_frag:
                    _body = sub(r"^CPRDAURUM_", "", aurum_frag)
                    output_names[_body.replace("_", "").upper()] = name
                if aurum_frag:
                    col_filter_set.add(f"BD_MEDI:{aurum_frag}")
            bd_filter = col_filter_set or None
            _use_exact_filter = True
        else:
            bd_filter = set(bd_list_entries)
            _use_exact_filter = False

    if config_preproc["filename"] in (None, "null", ""):
        config_preproc["filename"] = None
        filesToFormat = [config_preproc['filename_gold'],
                         config_preproc['filename_aurum'],]
        if config_preproc["filename_gold"][-3:] == "csv":
            config_preproc['filename_gold'] = f"{config_preproc['filename_gold'][:-4]}_formNulls.parquet"
            config_preproc['filename_aurum'] = f"{config_preproc['filename_aurum'][:-4]}_formNulls.parquet"
        elif config_preproc["filename_gold"][-7:] == "parquet":
            config_preproc['filename_gold'] = f"{config_preproc['filename_gold'][:-8]}_formNulls.parquet"
            config_preproc['filename_aurum'] = f"{config_preproc['filename_aurum'][:-8]}_formNulls.parquet"
        else:
            raise Exception("File type not recognised")
    else:
        filesToFormat = [config_preproc['filename']]
        if filesToFormat[0][-3:] == "csv":
            config_preproc["filename"] = f"{filesToFormat[0][:-4]}_formNulls.parquet"
        elif filesToFormat[0][-7:] == "parquet":
            config_preproc["filename"] = f"{filesToFormat[0][:-8]}_formNulls.parquet"
        else:
            raise Exception("File type not recognised")

    for file_ in filesToFormat:
        if file_[-3:] == "csv":
            # Read only the CSV header — no data loaded.
            with open(f"{dir_data}{file_}", "r", newline="") as _fh:
                _all_cols_raw = next(csv.reader(_fh))

            # Strip whitespace. CSV exports from some tools include a space after
            # each delimiter ("COL1, BD_MEDI:COND:1"), making csv.reader return
            # " BD_MEDI:COND:1" with a leading space that breaks startswith("BD_").
            # Polars scan_csv strips names automatically; we must do it explicitly here.
            _all_cols = [c.strip() for c in _all_cols_raw]

            # BD_ columns are the binary derived condition indicators used by the
            # pipeline.  B_MEDI: (single underscore) and B. columns are Dexter
            # intermediate outputs that are never read downstream — always exclude
            # them to roughly halve the working column count.
            _bd_cols_raw = [c for c in _all_cols if c.startswith("BD_")]
            _non_bd_cols = [c for c in _all_cols
                            if not c.startswith("BD_")
                            and not c.startswith("B_MEDI:")
                            and not c.startswith("B.")]

            if bd_filter is not None:
                if _use_exact_filter:
                    # Mapping file was used.  Normalise both the filter fragments
                    # and the candidate column names before comparing so that Gold
                    # fragments (e.g. "CPRD_ACTINIC_KERATOSIS") match Aurum-style
                    # column names (e.g. "BD_MEDI:CPRDAURUM_ACTINICKERATOSIS:2")
                    # and vice-versa.
                    _norm_filter = {_norm_bd_frag(f) for f in bd_filter}
                    _bd_cols_to_read = [
                        c for c in _bd_cols_raw
                        if _norm_bd_frag(sub(r":\d+$", "", c)) in _norm_filter
                    ]
                else:
                    # No mapping file: BD_LIST entries are direct column names.
                    # Try exact then fuzzy (legacy backward-compat path).
                    def _bd_matches(col: str) -> bool:
                        if col in bd_filter:
                            return True
                        stripped = sub(r":\d+$", "", col)
                        if stripped in bd_filter:
                            return True
                        col_body = sub(r"^BD_MEDI:", "", stripped)
                        for entry in bd_filter:
                            cond = entry[3:] if entry.startswith("BD_") else entry
                            if cond and col_body.replace("_", "").endswith(
                                cond.replace("_", "")
                            ):
                                return True
                        return False
                    _bd_cols_to_read = [c for c in _bd_cols_raw if _bd_matches(c)]

                if not _bd_cols_to_read:
                    logger.warning(
                        "BD_LIST filter matched 0 BD_ columns; "
                        "including all BD_ columns. Check that BD_LIST names "
                        "match the condition identifiers in the CSV."
                    )
                    _bd_cols_to_read = _bd_cols_raw

                logger.info(
                    f"  BD_ column selection: {len(_bd_cols_to_read)} of "
                    f"{len(_bd_cols_raw)} columns selected via BD_LIST"
                )
            else:
                _bd_cols_to_read = _bd_cols_raw

            _cols_to_read = _non_bd_cols + _bd_cols_to_read
            file_root_ = file_[:-4]

            change_colnames = {k: sub(r":\d+$", "", k)
                               for k in _cols_to_read if k.startswith("BD_MEDI:")}
            seen: set = set()
            cols_to_keep = []
            for col_name in _cols_to_read:
                renamed = change_colnames.get(col_name, col_name)
                if renamed not in seen:
                    seen.add(renamed)
                    cols_to_keep.append(col_name)
            change_colnames_kept = {k: v for k, v in change_colnames.items() if k in cols_to_keep}

            _checkpoint(
                f"formNulls: streaming CSV '{file_}' via PyArrow "
                f"({len(cols_to_keep)} cols: {len(_non_bd_cols)} cohort/demo + "
                f"{len(_bd_cols_to_read)} BD_)",
                logger,
            )
            _formNulls_csv_to_parquet(
                in_path=f"{dir_data}{file_}",
                out_path=f"{dir_data}{file_root_}_formNulls.parquet",
                include_columns=cols_to_keep,
                rename_map=change_colnames_kept,
                logger=logger,
            )
            _checkpoint("formNulls: CSV→parquet complete", logger)
            _free_memory()
            _checkpoint("formNulls: memory released", logger)

        elif file_[-7:] == "parquet":
            dat = pl.scan_parquet(f"{dir_data}{file_}")
            col_names_for_nulls = dat.collect_schema().names()
            file_root_ = file_[:-8]

            dat = dat.with_columns([
                pl.when(pl.col(c) == "")
                  .then(pl.lit(None, dtype=pl.String))
                  .otherwise(pl.col(c))
                  .alias(c)
                for c in col_names_for_nulls
            ])

            change_colnames = {k: sub(r":\d+$", "", k)
                               for k in col_names_for_nulls if k.startswith("BD_MEDI:")}
            seen: set = set()
            cols_to_keep = []
            for col_name in col_names_for_nulls:
                renamed = change_colnames.get(col_name, col_name)
                if renamed not in seen:
                    seen.add(renamed)
                    cols_to_keep.append(col_name)
            change_colnames_kept = {k: v for k, v in change_colnames.items() if k in cols_to_keep}
            dat = dat.select(cols_to_keep).rename(change_colnames_kept)

            _checkpoint(f"formNulls: starting sink_parquet → {file_root_}_formNulls.parquet", logger)
            dat.sink_parquet(f"{dir_data}{file_root_}_formNulls.parquet", row_group_size=100_000)
            _checkpoint("formNulls: sink_parquet complete", logger)
            del dat
            _free_memory()
            _checkpoint("formNulls: memory released", logger)
        else:
            raise Exception("File type not recognised")
    logger.info("    Formatting null values finished")

    ###LinkingAurumGold############################################################
    if config_preproc["filename"] in (None, "null", ""):
        print("Linking")
        logger.info("Linking Gold and Aurum")

        dat_a = config_preproc['filename_gold']
        dat_b = config_preproc['filename_aurum']

        outFile="dat_linked.parquet"
        rmDup(
            dat_a,
            dat_b,
            A_ind=0,
            B_ind=2,
            map_file=f"{dir_data}{config_preproc['map_file_AtoB']}",
            map_delim=config_preproc['map_delim_AtoB'],
            low_memory=False,
            wdir=dir_data,
            logger=logger,
            outFile=outFile,
            )
        logger.info("    Linking finished")
        flag_temp_file = True
    else:
        outFile = config_preproc["filename"]

    ###CoalesceSourceBDCols########################################################
    # Coalesce source-prefixed BD_MEDI: columns into one column per condition.
    # When a condition_map_file was supplied, output_names maps each condition's
    # normalised key to its Paper Short Name, which becomes the column name in the
    # processed parquet.  Without a mapping file the existing canonical_name logic
    # derives a BD_CONDNAME from the Gold column name.
    print("Coalescing BD source columns")
    logger.info("Coalescing BD_ source columns")
    _coalesced_path = f"{dir_data}dat_coalesced.parquet"
    coalesce_bd_source_cols(f"{dir_data}{outFile}", _coalesced_path, logger, output_names=output_names)
    if flag_temp_file:
        os.remove(f"{dir_data}{outFile}")
    flag_temp_file = True
    outFile = "dat_coalesced.parquet"
    _checkpoint("Coalescing BD_ source columns complete", logger)

    ###LinkHes#####################################################################
    if config_preproc["path_hes"] is not None:
        print("Linking Hes")
        logger.info("Linking HES")

        link_hes(
                f"{dir_data}{outFile}",
                config_preproc["path_hes"],
                config_preproc["col_patid_cprd"],
                config_preproc["col_patid_hes"],
                f"{dir_data}dat_hesLinked.parquet",
                low_memory=True,
                )
        if flag_temp_file:
            os.remove(f"{dir_data}{outFile}")
        else:
            flag_temp_file = True
        outFile = "dat_hesLinked.parquet"

        # In file = Paths.DAT_AURGOLD
        # write to Paths.DAT_HES_LINK
        logger.info("   Linking HES finished")

    ###MergeCols#####################################################################
    if outFile.find(".csv") != -1:
        file_type = "csv"
    else:
        file_type = "parquet"

    if config_preproc["mergeCols_AtoB"] is not None:
        print("Merging Cols")
        logger.info("Merging columns")

        mergeCols(
                dir_data,
                outFile,
                config_preproc["mergeCols_AtoB"],
                file_type = file_type,
                low_memory = True,
                logger=logger,
                outFile="condMerged.parquet",
                date_fmt=date_fmt,
                rm_old_cols=config_preproc["rm_old_cols"],
                )
        if flag_temp_file:
            os.remove(f"{dir_data}{outFile}")
        else:
            flag_temp_file = True
        outFile = "condMerged.parquet"
        logger.info("    Merging Cols finished")

    ###CombineLevels#####################################################################
    if outFile.find(".csv") != -1:
        file_type = "csv"
    else:
        file_type = "parquet"

    if config_preproc["combineLevels"] is not None:
        print("Processing Column Levels")
        logger.info("Combining levels")

        combineLevels(dir_data,
                      outFile,
                      config_preproc["combineLevels"],
                      file_type=file_type,
                      outFile="dat_updatedLevels.parquet",
                      )
        if flag_temp_file:
            os.remove(f"{dir_data}{outFile}")
        else:
            flag_temp_file = True
        outFile = "dat_updatedLevels.parquet"
        logger.info("    Processing Column Levels finished")

    ###LinkImd#####################################################################
    if outFile.find(".csv") != -1:
        is_parquet = False
    else:
        is_parquet = True

    if config_preproc['link_imd']:
        print("Linking IMD")
        logger.info("Linking IMD")

        process_imd(
                outFile,
                dir_data,
                file_map = config_preproc["imd_map_file"],
                low_memory=True,
                is_parquet=is_parquet,
                logger=logger,
                outFile="dat_processed.parquet",
                )
        if flag_temp_file:
            os.remove(f"{dir_data}{outFile}")
        else:
            flag_temp_file = True
        outFile="dat_processed.parquet"
        logger.info("    Linking IMD finished")

    os.rename(f"{dir_data}{outFile}", f"{dir_data}dat_processed.parquet")
    _checkpoint("dat_processed.parquet written", logger)

    ###DerivedColumns##############################################################
    derived_cfg = config_preproc.get("derived_columns") or {}
    any_enabled = any(
        isinstance(v, dict) and v.get("enabled")
        for v in derived_cfg.values()
    )
    if any_enabled:
        logger.info("Deriving extra columns (IMD quintile / age binary)")
        print("Deriving extra columns")
        temp_path = f"{dir_data}dat_derived_temp.parquet"
        derive_columns(
            in_path=f"{dir_data}dat_processed.parquet",
            out_path=temp_path,
            derived_config=derived_cfg,
        )
        os.remove(f"{dir_data}dat_processed.parquet")
        os.rename(temp_path, f"{dir_data}dat_processed.parquet")
        logger.info("    Deriving extra columns finished")

    ###CreateBatchFiles############################################################
    _checkpoint("create_batch_files check", logger)
    if config_incprev is not None and config_incprev.get("create_batch_files"):
        logger.info("Creating per-batch parquet files")
        print("Creating per-batch parquet files (column-wise batching)")

        processed_path = f"{dir_data}dat_processed.parquet"

        # Discover BD_ columns from file if not specified in config
        if config_incprev.get("BD_LIST"):
            bd_list = list(config_incprev["BD_LIST"])
        else:
            bd_list = [
                c for c in pl.scan_parquet(processed_path).collect_schema().names()
                if c.startswith("BD_")
            ]

        # Flatten DEMOGRAPHY (may contain nested lists for composite groups)
        demo_cols = list(set(
            item
            for entry in config_incprev.get("DEMOGRAPHY", [])
            for item in (entry if isinstance(entry, list) else [entry])
        ))

        core_cols = [
            config_incprev.get("col_index_date", "INDEX_DATE"),
            config_incprev.get("col_end_date", "END_DATE"),
        ]

        _checkpoint(f"create_batch_parquet_files: {len(bd_list)} conditions, batch_size={config_incprev.get('batch_size', 10)}", logger)
        create_batch_parquet_files(
            in_path=processed_path,
            out_dir=dir_data,
            bd_list=bd_list,
            batch_size=config_incprev.get("batch_size", 10),
            core_cols=core_cols,
            demo_cols=demo_cols,
        )
        _checkpoint("create_batch_parquet_files complete", logger)
        logger.info("    Creating per-batch parquet files finished")

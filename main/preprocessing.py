import sys
import os
import datetime
import csv
import multiprocessing as mp
from itertools import repeat
import logging
from re import sub
import yaml
import resource

import polars as pl
import pyarrow.dataset as ds
from main.preprocessing_functions import process_imd, rmDup, mergeCols, combineLevels, link_hes, create_batch_parquet_files, derive_columns

def _mem_gb() -> str:
    """Return current RSS memory usage as a formatted string."""
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports in kB; macOS in bytes
    rss_gb = rss_bytes / (1024 ** 2) if sys.platform != "darwin" else rss_bytes / (1024 ** 3)
    return f"{rss_gb:.1f} GB"


def _checkpoint(label: str, logger) -> None:
    msg = f"[MEM {_mem_gb()}] {label}"
    print(msg, flush=True)
    logger.info(msg)


def _csv_to_parquet_streaming(
        csv_path: str,
        out_path: str,
        bd_filter: "set | None",
        logger,
        block_size_mb: int = 64,
) -> None:
    """Convert a wide CSV to parquet by streaming through PyArrow in chunks.

    Uses PyArrow's open_csv (chunked reader) so that at most one block of rows
    is in RAM at a time, regardless of how many columns the file has.

    Empty strings are replaced with null at parse time via ConvertOptions.
    When bd_filter is set, only BD_ columns matching the filter are read;
    all non-BD_ columns are always included.  BD_MEDI:COND:N column names are
    stripped to BD_MEDI:COND and duplicate post-rename names are deduplicated
    (first occurrence wins).
    """
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    import pyarrow.parquet as pq

    # Read only the header line — no data loaded.
    with open(csv_path, "r", newline="") as fh:
        all_cols_raw = next(csv.reader(fh))

    # Strip whitespace from every column name.  CSV files exported from some
    # tools include a space after each delimiter (e.g. "COL1, BD_MEDI:COND:1"),
    # which makes csv.reader return " BD_MEDI:COND:1" — a leading space that
    # silently breaks startswith("BD_").  PyArrow must receive the raw (un-stripped)
    # name to match the literal CSV header, so we keep both forms.
    all_cols = [c.strip() for c in all_cols_raw]
    # Map stripped → first matching raw name (for PyArrow include_columns).
    stripped_to_raw: dict = {}
    for raw, stripped in zip(all_cols_raw, all_cols):
        if stripped not in stripped_to_raw:
            stripped_to_raw[stripped] = raw

    bd_cols_raw = [c for c in all_cols if c.startswith("BD_")]
    non_bd_cols = [c for c in all_cols if not c.startswith("BD_")]

    if bd_filter is not None:
        bd_cols_to_read = [
            c for c in bd_cols_raw
            if c in bd_filter or sub(r":\d+$", "", c) in bd_filter
        ]
        logger.info(
            f"  BD_ column selection: {len(bd_cols_to_read)} of "
            f"{len(bd_cols_raw)} columns selected via BD_LIST"
        )
    else:
        bd_cols_to_read = bd_cols_raw

    cols_to_read = non_bd_cols + bd_cols_to_read

    _checkpoint(
        f"formNulls: streaming CSV '{os.path.basename(csv_path)}' "
        f"({len(cols_to_read)} cols: {len(non_bd_cols)} cohort/demo + "
        f"{len(bd_cols_to_read)} BD_)",
        logger,
    )

    # Compute rename mapping (strip BD_MEDI: numeric Dexter suffix).
    # All names here are already stripped of whitespace.
    change_colnames = {k: sub(r":\d+$", "", k)
                       for k in cols_to_read if k.startswith("BD_MEDI:")}

    # Dedup: keep only the first occurrence of each post-rename column name.
    seen: set = set()
    final_cols_stripped: list = []
    for c in cols_to_read:
        renamed = change_colnames.get(c, c)
        if renamed not in seen:
            seen.add(renamed)
            final_cols_stripped.append(c)

    # PyArrow include_columns / column_types must use the raw (un-stripped) names
    # that appear literally in the CSV header.
    final_cols_raw = [stripped_to_raw[c] for c in final_cols_stripped]
    final_cols_renamed = [change_colnames.get(c, c) for c in final_cols_stripped]

    # Force every selected column to string type; replace "" with null at parse time.
    # This avoids both Polars streaming limitations and PyArrow's own type inference
    # (which could silently coerce patient IDs or date strings to numeric types).
    col_types = {c: pa.string() for c in final_cols_raw}
    convert_opts = pa_csv.ConvertOptions(
        strings_can_be_null=True,
        null_values=[""],
        column_types=col_types,
        include_columns=final_cols_raw,
    )
    read_opts = pa_csv.ReadOptions(block_size=block_size_mb * 1024 * 1024)

    reader = pa_csv.open_csv(csv_path, read_options=read_opts, convert_options=convert_opts)

    writer = None
    try:
        for batch in reader:
            renamed = batch.rename_columns(final_cols_renamed)
            if writer is None:
                writer = pq.ParquetWriter(out_path, renamed.schema)
            writer.write_batch(renamed)
    finally:
        if writer:
            writer.close()


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

    flag_temp_file: bool = False

    _checkpoint("preprocessing() started", logger)

    ## Format Null ################################################################
    logger.info("Formatting null values")

    # Build a set of BD_ column names actually needed (from BD_LIST in incprev config).
    # When the raw CSV has hundreds of condition columns but only a few are needed,
    # reading only those columns avoids loading the entire wide file into RAM.
    bd_filter: set | None = None
    if config_incprev and config_incprev.get("BD_LIST"):
        bd_filter = set(str(b) for b in config_incprev["BD_LIST"])

    if config_preproc["filename"] is None:
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
            file_root_ = file_[:-4]
            out_parquet = f"{dir_data}{file_root_}_formNulls.parquet"
            _csv_to_parquet_streaming(
                csv_path=f"{dir_data}{file_}",
                out_path=out_parquet,
                bd_filter=bd_filter,
                logger=logger,
            )
            _checkpoint(f"formNulls: {file_} → parquet complete", logger)

        elif file_[-7:] == "parquet":
            # Parquet inputs are already typed; apply null replacement via Polars streaming.
            file_root_ = file_[:-8]
            dat = pl.scan_parquet(f"{dir_data}{file_}")
            col_names_for_nulls = dat.collect_schema().names()

            dat = dat.with_columns([
                pl.when(pl.col(c) == "")
                  .then(pl.lit(None, dtype=pl.String))
                  .otherwise(pl.col(c))
                  .alias(c)
                for c in col_names_for_nulls
            ])

            change_colnames = {k: sub(r":\d+$", "", k)
                               for k in col_names_for_nulls if k.startswith("BD_MEDI:")}
            seen_p: set = set()
            cols_to_keep = []
            for col_name in col_names_for_nulls:
                renamed = change_colnames.get(col_name, col_name)
                if renamed not in seen_p:
                    seen_p.add(renamed)
                    cols_to_keep.append(col_name)
            change_colnames_kept = {k: v for k, v in change_colnames.items() if k in cols_to_keep}
            dat = dat.select(cols_to_keep).rename(change_colnames_kept)

            _checkpoint(f"formNulls: starting sink_parquet → {file_root_}_formNulls.parquet", logger)
            dat.sink_parquet(f"{dir_data}{file_root_}_formNulls.parquet")
            _checkpoint("formNulls: sink_parquet complete", logger)
        else:
            raise Exception("File type not recognised")
    logger.info("    Formatting null values finished")

    ###LinkingAurumGold############################################################
    if config_preproc["filename"] is None:
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

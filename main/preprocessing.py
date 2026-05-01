import sys
import os
import datetime
import csv
import multiprocessing as mp
from itertools import repeat
import logging
from re import sub
import yaml

import polars as pl
import pyarrow.dataset as ds
from main.preprocessing_functions import process_imd, rmDup, mergeCols, combineLevels, link_hes, create_batch_parquet_files, derive_columns

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

    flag_temp_file: bool = False

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
            # Read only the CSV header to determine which columns exist — no data loaded.
            with open(f"{dir_data}{file_}", "r", newline="") as _fh:
                _all_cols = next(csv.reader(_fh))

            # Separate BD_ columns from cohort/demographic columns.
            _bd_cols_raw = [c for c in _all_cols if c.startswith("BD_")]
            _non_bd_cols = [c for c in _all_cols if not c.startswith("BD_")]

            # When BD_LIST is configured, read only those BD_ columns.
            # BD_MEDI: columns carry a numeric Dexter suffix (e.g. BD_MEDI:COND:1);
            # match against the stripped name (BD_MEDI:COND) as well as the raw name.
            if bd_filter is not None:
                _bd_cols_to_read = [
                    c for c in _bd_cols_raw
                    if c in bd_filter or sub(r":\d+$", "", c) in bd_filter
                ]
                logger.info(
                    f"  BD_ column selection: {len(_bd_cols_to_read)} of "
                    f"{len(_bd_cols_raw)} columns selected via BD_LIST"
                )
            else:
                _bd_cols_to_read = _bd_cols_raw

            _cols_to_read = _non_bd_cols + _bd_cols_to_read
            dat = pl.scan_csv(
                f"{dir_data}{file_}",
                infer_schema_length=0,
                columns=_cols_to_read,
            )
            col_names_for_nulls = _cols_to_read  # all read as String with infer_schema_length=0
            file_root_ = file_[:-4]

        elif file_[-7:] == "parquet":
            dat = pl.scan_parquet(f"{dir_data}{file_}")
            col_names_for_nulls = dat.collect_schema().names()
            file_root_ = file_[:-8]
        else:
            raise Exception("File type not recognised")

        # Replace empty strings with null using explicit per-column expressions.
        # pl.all() inside when/then/otherwise is not guaranteed to be streamable
        # in Polars and can cause a silent full-collect fallback, loading the
        # entire dataset into RAM.  Explicit named columns avoid this.
        dat = dat.with_columns([
            pl.when(pl.col(c) == "")
              .then(pl.lit(None, dtype=pl.String))
              .otherwise(pl.col(c))
              .alias(c)
            for c in col_names_for_nulls
        ])

        # rm numeric suffix from Dexter out (assumes unique codelist names)
        change_colnames = {k: sub(r":\d+$", "", k)
                           for k in col_names_for_nulls if k.startswith("BD_MEDI:")}

        # Drop columns that would become duplicates after renaming (keep first occurrence)
        seen: set = set()
        cols_to_keep = []
        for col_name in col_names_for_nulls:
            renamed = change_colnames.get(col_name, col_name)
            if renamed not in seen:
                seen.add(renamed)
                cols_to_keep.append(col_name)

        change_colnames_kept = {k: v for k, v in change_colnames.items() if k in cols_to_keep}
        dat = dat.select(cols_to_keep).rename(change_colnames_kept)

        dat.sink_parquet(f"{dir_data}{file_root_}_formNulls.parquet")
    logger.info("    Formatting null values finished")
    del dat

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

        create_batch_parquet_files(
            in_path=processed_path,
            out_dir=dir_data,
            bd_list=bd_list,
            batch_size=config_incprev.get("batch_size", 10),
            core_cols=core_cols,
            demo_cols=demo_cols,
        )
        logger.info("    Creating per-batch parquet files finished")

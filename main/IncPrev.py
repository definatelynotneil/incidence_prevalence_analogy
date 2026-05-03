import os
import datetime
import csv
from itertools import repeat
import multiprocessing as mp
from re import match, compile, sub
from typing import Optional
import pyarrow.dataset as ds
import polars as pl
from main.ANALOGY_SCIENTIFIC.IncPrevMethods_polars import IncPrev


def _resolve_bd_list(bd_list: list, filename: str) -> list:
    """Map BD_LIST entries to actual column names in the data file.

    Preprocessing coalesces source-prefixed columns (BD_MEDI:CPRD_*, etc.)
    into clean BD_CONDNAME columns, so exact matches are expected for
    well-formed BD_LIST entries.  Fuzzy matching is retained as a fallback
    for data files that have not been through the coalescing step.
    """
    if filename.endswith(".parquet"):
        actual_bd = [c for c in ds.dataset(filename, format="parquet").schema.names
                     if c.startswith("BD_")]
    elif filename.endswith(".csv"):
        with open(filename, "r", encoding="utf8") as _f:
            actual_bd = [c.strip() for c in next(csv.reader(_f))
                         if c.strip().startswith("BD_")]
    else:
        return bd_list

    actual_bd_set = set(actual_bd)
    resolved = []
    for entry in bd_list:
        # Exact match (expected case after preprocessing coalescing)
        if entry in actual_bd_set:
            resolved.append(entry)
            continue
        # Fuzzy fallback: normalise underscores and strip BD_MEDI: source prefix
        cond = entry[3:] if entry.startswith("BD_") else entry
        cond_norm = cond.replace("_", "").upper()
        # Prefer source-prefixed columns (CPRD_ etc.) over bare BD_MEDI: columns
        # to avoid resolving to a non-existent bare name.
        hit = next(
            (col for col in actual_bd
             if "CPRD" in col
             and sub(r"^BD_MEDI:[A-Z0-9]+_", "", col)
                .replace("_", "").upper() == cond_norm),
            None,
        )
        if hit is None:
            hit = next(
                (col for col in actual_bd
                 if sub(r"^BD_MEDI:", "", col)
                    .replace("_", "").upper() == cond_norm),
                None,
            )
        resolved.append(hit if hit is not None else entry)
    return resolved


def processBatch(
    batch,
    STUDY_START_DATE,
    STUDY_END_DATE,
    FILENAME,
    DEMOGRAPHY,
    col_end_date,
    col_index_date,
    date_fmt,
    dir_out,
    batchId,
    streaming_chunk_size: Optional[int] = None,
) -> None:
    # Get unique individual demographic columns
    CATGS = list(set([
        item if isinstance(sublist, str) else item
        for sublist in DEMOGRAPHY
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]))

    if isinstance(batch, str):
        batch = [batch]
    else:
        batch = list(batch)

    cols = [col_index_date, col_end_date] + batch
    if CATGS:
        cols = cols + CATGS

    common_kwargs = dict(
        BASELINE_DATE_LIST=batch,
        DEMOGRAPHY=DEMOGRAPHY,
        cols=cols,
        col_end_date=col_end_date,
        col_index_date=col_index_date,
        date_fmt=date_fmt,
        verbose=False,
    )

    # Incidence
    dat_inc = IncPrev(STUDY_END_DATE[0], STUDY_START_DATE[0], FILENAME, **common_kwargs)
    results_inc = dat_inc.runAnalysis(inc=True, prev=False, streaming_chunk_size=streaming_chunk_size)[0]

    # Prevalence
    dat_prev = IncPrev(STUDY_END_DATE[1], STUDY_START_DATE[1], FILENAME, **common_kwargs)
    results_prev = dat_prev.runAnalysis(inc=False, prev=True, streaming_chunk_size=streaming_chunk_size)[1]

    for result_ in (results_inc, results_prev):
        metric = "prev" if "Prevalence" in result_.columns else "inc"
        result_.write_csv(f"{dir_out}out_{metric}_{batchId}.csv")


def run_incprev(conf_incprev: dict,
                dir_data: str,
                dir_out: str,
                date_fmt: str) -> None:

    FULL_FILENAME = f"{dir_data}{conf_incprev['filename']}"
    streaming_chunk_size: Optional[int] = conf_incprev.get("streaming_chunk_size")
    create_batch_files: bool = conf_incprev.get("create_batch_files", False)

    STUDY_START_DATE_INC = datetime.datetime(
        year=conf_incprev["start_date"]["inc"]["year"],
        month=conf_incprev["start_date"]["inc"]["month"],
        day=conf_incprev["start_date"]["inc"]["day"],
    )
    STUDY_END_DATE_INC = datetime.datetime(
        year=conf_incprev["end_date"]["inc"]["year"],
        month=conf_incprev["end_date"]["inc"]["month"],
        day=conf_incprev["end_date"]["inc"]["day"],
    )
    STUDY_START_DATE_PREV = datetime.datetime(
        year=conf_incprev["start_date"]["prev"]["year"],
        month=conf_incprev["start_date"]["prev"]["month"],
        day=conf_incprev["start_date"]["prev"]["day"],
    )
    STUDY_END_DATE_PREV = datetime.datetime(
        year=conf_incprev["end_date"]["prev"]["year"],
        month=conf_incprev["end_date"]["prev"]["month"],
        day=conf_incprev["end_date"]["prev"]["day"],
    )
    STUDY_START_DATE = [STUDY_START_DATE_INC, STUDY_START_DATE_PREV]
    STUDY_END_DATE = [STUDY_END_DATE_INC, STUDY_END_DATE_PREV]

    # Discover condition (BD_) columns
    if conf_incprev["BD_LIST"] is None:
        if FULL_FILENAME.endswith(".parquet"):
            dataset = ds.dataset(FULL_FILENAME, format="parquet")
            col_head = list(dataset.head(1).to_pylist()[0].keys())
            del dataset
        elif FULL_FILENAME.endswith(".csv"):
            with open(FULL_FILENAME, "r", encoding="utf8") as f:
                col_head = next(csv.reader(f))
        else:
            raise Exception("Cannot determine file type from extension")
        BASELINE_DATE_LIST = [c for c in col_head if c.startswith("BD_")]
    else:
        BASELINE_DATE_LIST = _resolve_bd_list(
            list(conf_incprev["BD_LIST"]), FULL_FILENAME
        )

    batch_size = conf_incprev["batch_size"]

    # Split conditions into batches
    batched_bd = [
        tuple(BASELINE_DATE_LIST[i : i + batch_size])
        for i in range(0, len(BASELINE_DATE_LIST), batch_size)
    ]

    batch_ids = list(range(len(batched_bd)))

    # Choose per-batch file if column-wise batch files were created
    def filename_for_batch(batch_id: int) -> str:
        candidate = f"{dir_data}dat_batch_{batch_id}.parquet"
        if create_batch_files and os.path.exists(candidate):
            return candidate
        return FULL_FILENAME

    batches = [
        (
            bd_batch,
            STUDY_START_DATE,
            STUDY_END_DATE,
            filename_for_batch(batch_id),
            conf_incprev["DEMOGRAPHY"],
            conf_incprev["col_end_date"],
            conf_incprev["col_index_date"],
            date_fmt,
            dir_out,
            batch_id,
            streaming_chunk_size,
        )
        for bd_batch, batch_id in zip(batched_bd, batch_ids)
    ]

    N_PROCESSES = conf_incprev.get("n_processes") or 1
    if N_PROCESSES == 1:
        for batch_ in batches:
            processBatch(*batch_)
    else:
        with mp.get_context("spawn").Pool(processes=N_PROCESSES) as pool:
            pool.starmap(processBatch, batches)

    # Concatenate per-batch output files into final CSVs
    files_out = os.listdir(dir_out)
    pattern_inc = compile(r".*inc_[0-9].*")
    pattern_prev = compile(r".*prev_[0-9].*")
    file_names_inc = sorted([x for x in files_out if match(pattern_inc, x)])
    file_names_prev = sorted([x for x in files_out if match(pattern_prev, x)])

    def write_out(file_names, output_file, dir_):
        with open(f"{dir_}{output_file}", "w") as outfile:
            for i, file_name in enumerate(file_names):
                with open(f"{dir_}{file_name}", "r") as infile:
                    if i != 0:
                        next(infile)  # skip header after first file
                    outfile.write(infile.read())

    write_out(file_names_inc, "inc_crude.csv", dir_out)
    write_out(file_names_prev, "prev_crude.csv", dir_out)

    for f in file_names_inc + file_names_prev:
        os.remove(f"{dir_out}{f}")

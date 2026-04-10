"""
tests/test_streaming.py
========================
Verify that the streaming (row-wise chunked) incidence/prevalence calculation
produces numerically identical results to the standard in-memory path.

The test generates synthetic data, writes it to a temporary parquet file, runs
both paths, and asserts that numerators and denominators match exactly for every
(Condition, Group, Subgroup, Date) combination.
"""
import os
import shutil
import pytest
import polars as pl
from tests.make_test_dat import get_example_dat
from main.ANALOGY_SCIENTIFIC.IncPrevMethods_polars import IncPrev
import datetime


STUDY_START_INC = datetime.datetime(2001, 1, 1)
STUDY_END_INC = datetime.datetime(2003, 12, 31)
STUDY_START_PREV = datetime.datetime(2001, 7, 1)
STUDY_END_PREV = datetime.datetime(2003, 12, 31)

DEMOGRAPHY = [
    "pat_catg_a",
    ["AGE_CATEGORY", "SEX"],
]

DATE_FMT = "%d-%m-%Y"


@pytest.fixture(scope="module")
def tmp_parquet(tmp_path_factory):
    """Write synthetic test data as parquet and return the path."""
    d = tmp_path_factory.mktemp("data")
    path = str(d / "test_dat.parquet")
    get_example_dat(2_000).write_parquet(path)
    return path


def _run_incprev(filename, streaming_chunk_size, is_inc):
    conditions = ["condition_a", "condition_b"]
    catgs = ["pat_catg_a", "AGE_CATEGORY", "SEX"]
    cols = ["INDEX_DATE", "END_DATE"] + conditions + catgs

    study_end = STUDY_END_INC if is_inc else STUDY_END_PREV
    study_start = STUDY_START_INC if is_inc else STUDY_START_PREV

    obj = IncPrev(
        study_end,
        study_start,
        filename,
        conditions,
        DEMOGRAPHY,
        cols,
        date_fmt=DATE_FMT,
    )
    inc_res, prev_res = obj.runAnalysis(
        inc=is_inc,
        prev=not is_inc,
        streaming_chunk_size=streaming_chunk_size,
    )
    return inc_res if is_inc else prev_res


def _compare(standard: pl.DataFrame, streamed: pl.DataFrame, label: str):
    key_cols = ["Condition", "Group", "Subgroup", "Date"]
    std_sorted = standard.sort(key_cols)
    str_sorted = streamed.sort(key_cols)

    assert std_sorted.shape == str_sorted.shape, (
        f"{label}: row count mismatch: standard={std_sorted.shape[0]}, "
        f"streaming={str_sorted.shape[0]}"
    )
    for col in ("Numerator", "Denominator"):
        diff = (std_sorted[col] - str_sorted[col]).abs().max()
        assert diff < 1e-6, f"{label} {col}: max difference {diff}"


@pytest.mark.parametrize("chunk_size", [50, 500, 5000])
def test_streaming_incidence_matches_standard(tmp_parquet, chunk_size):
    standard = _run_incprev(tmp_parquet, None, is_inc=True)
    streamed = _run_incprev(tmp_parquet, chunk_size, is_inc=True)
    _compare(standard, streamed, f"Incidence chunk={chunk_size}")


@pytest.mark.parametrize("chunk_size", [50, 500, 5000])
def test_streaming_prevalence_matches_standard(tmp_parquet, chunk_size):
    standard = _run_incprev(tmp_parquet, None, is_inc=False)
    streamed = _run_incprev(tmp_parquet, chunk_size, is_inc=False)
    _compare(standard, streamed, f"Prevalence chunk={chunk_size}")


def test_streaming_no_demography(tmp_parquet):
    """Streaming with no DEMOGRAPHY should return only Overall rows."""
    conditions = ["condition_a"]
    cols = ["INDEX_DATE", "END_DATE"] + conditions

    obj = IncPrev(
        STUDY_END_INC,
        STUDY_START_INC,
        tmp_parquet,
        conditions,
        [],
        cols,
        date_fmt=DATE_FMT,
    )
    inc_stream, _ = obj.runAnalysis(inc=True, prev=False, streaming_chunk_size=100)
    obj2 = IncPrev(
        STUDY_END_INC,
        STUDY_START_INC,
        tmp_parquet,
        conditions,
        [],
        cols,
        date_fmt=DATE_FMT,
    )
    inc_std, _ = obj2.runAnalysis(inc=True, prev=False, streaming_chunk_size=None)

    assert set(inc_stream["Group"].unique()) == {"Overall"}
    _compare(inc_std, inc_stream, "No-demography incidence")

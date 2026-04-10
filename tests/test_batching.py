"""
tests/test_batching.py
=======================
Tests for column-wise batch parquet file creation.
"""
import os
import polars as pl
import pytest
from tests.make_test_dat import get_example_dat
from main.preprocessing_functions import create_batch_parquet_files


@pytest.fixture(scope="module")
def source_parquet(tmp_path_factory):
    d = tmp_path_factory.mktemp("batch_data")
    path = str(d / "source.parquet")
    df = get_example_dat(500)
    # Add two fake BD_ columns
    df = df.with_columns([
        pl.lit("2001-01-01").alias("BD_cond_a"),
        pl.lit("2002-06-15").alias("BD_cond_b"),
        pl.lit("2000-03-01").alias("BD_cond_c"),
    ])
    df.write_parquet(path)
    return path, str(d) + "/"


def test_batch_files_created(source_parquet):
    path, out_dir = source_parquet
    bd_list = ["BD_cond_a", "BD_cond_b", "BD_cond_c"]
    create_batch_parquet_files(
        in_path=path,
        out_dir=out_dir,
        bd_list=bd_list,
        batch_size=2,
        core_cols=["INDEX_DATE", "END_DATE"],
        demo_cols=["AGE_CATEGORY", "SEX"],
    )
    # batch_size=2 with 3 BD_ cols → 2 batch files
    assert os.path.exists(f"{out_dir}dat_batch_0.parquet")
    assert os.path.exists(f"{out_dir}dat_batch_1.parquet")


def test_batch_0_has_correct_columns(source_parquet):
    _, out_dir = source_parquet
    df = pl.read_parquet(f"{out_dir}dat_batch_0.parquet")
    assert "INDEX_DATE" in df.columns
    assert "END_DATE" in df.columns
    assert "AGE_CATEGORY" in df.columns
    assert "BD_cond_a" in df.columns
    assert "BD_cond_b" in df.columns
    # batch 0 should NOT have cond_c (it's in batch 1)
    assert "BD_cond_c" not in df.columns


def test_batch_1_has_correct_columns(source_parquet):
    _, out_dir = source_parquet
    df = pl.read_parquet(f"{out_dir}dat_batch_1.parquet")
    assert "BD_cond_c" in df.columns
    assert "BD_cond_a" not in df.columns


def test_batch_row_counts_match_source(source_parquet):
    path, out_dir = source_parquet
    source_rows = pl.read_parquet(path).shape[0]
    batch0_rows = pl.read_parquet(f"{out_dir}dat_batch_0.parquet").shape[0]
    batch1_rows = pl.read_parquet(f"{out_dir}dat_batch_1.parquet").shape[0]
    assert batch0_rows == source_rows
    assert batch1_rows == source_rows


def test_batch_missing_demo_col_skipped(source_parquet):
    """A demo column not in the source file should be silently excluded."""
    path, out_dir = source_parquet
    create_batch_parquet_files(
        in_path=path,
        out_dir=out_dir,
        bd_list=["BD_cond_a"],
        batch_size=1,
        core_cols=["INDEX_DATE", "END_DATE"],
        demo_cols=["AGE_CATEGORY", "NONEXISTENT_COL"],
    )
    df = pl.read_parquet(f"{out_dir}dat_batch_0.parquet")
    assert "NONEXISTENT_COL" not in df.columns
    assert "AGE_CATEGORY" in df.columns

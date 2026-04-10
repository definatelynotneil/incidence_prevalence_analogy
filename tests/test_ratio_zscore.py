"""
tests/test_ratio_zscore.py
===========================
Unit tests for the statistical functions in main/ratioZscore.py:
  - Geometric mean ignores zeros and NaNs
  - SE of log-ratio uses the Dobson formula correctly
  - Z-score and CI are consistent with the SE
  - run_ratio_zscore() produces expected output structure on minimal synthetic data
"""
import os
import shutil
import math
import numpy as np
import pandas as pd
import pytest

from main.ratioZscore import (
    _geom_mean,
    _se_log_ratio,
    _ratio_ci_and_z,
    run_ratio_zscore,
)


# ---------------------------------------------------------------------------
# _geom_mean
# ---------------------------------------------------------------------------

def test_geom_mean_basic():
    s = pd.Series([1.0, 2.0, 4.0])
    expected = math.exp((math.log(1) + math.log(2) + math.log(4)) / 3)
    assert abs(_geom_mean(s) - expected) < 1e-10


def test_geom_mean_ignores_zeros():
    s = pd.Series([0.0, 2.0, 4.0])
    expected = math.exp((math.log(2) + math.log(4)) / 2)
    assert abs(_geom_mean(s) - expected) < 1e-10


def test_geom_mean_all_zeros():
    assert math.isnan(_geom_mean(pd.Series([0.0, 0.0])))


def test_geom_mean_empty():
    assert math.isnan(_geom_mean(pd.Series([], dtype=float)))


# ---------------------------------------------------------------------------
# _se_log_ratio
# ---------------------------------------------------------------------------

def test_se_log_ratio_known_value():
    # se = sqrt(var/rate^2 + refvar/refrate^2)
    var, rate, refvar, refrate = 4.0, 2.0, 9.0, 3.0
    expected = math.sqrt(4.0 / 4.0 + 9.0 / 9.0)  # = sqrt(2)
    assert abs(_se_log_ratio(var, rate, refvar, refrate) - expected) < 1e-10


def test_se_log_ratio_zero_rate():
    assert math.isnan(_se_log_ratio(1.0, 0.0, 1.0, 1.0))


def test_se_log_ratio_negative_rate():
    assert math.isnan(_se_log_ratio(1.0, -1.0, 1.0, 1.0))


def test_se_log_ratio_nan_inputs():
    assert math.isnan(_se_log_ratio(np.nan, 1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# _ratio_ci_and_z
# ---------------------------------------------------------------------------

def test_ratio_ci_and_z_unity_ratio():
    """log(1) = 0, so z-score should be 0 and CI should be symmetric."""
    z, lo, hi = _ratio_ci_and_z(1.0, se_log=0.1)
    assert abs(z) < 1e-10
    assert lo < 1.0 < hi
    # CI should be symmetric on log scale
    assert abs(math.log(hi) + math.log(lo)) < 1e-10


def test_ratio_ci_and_z_high_ratio():
    """ratio > 1 → positive z-score."""
    z, lo, hi = _ratio_ci_and_z(2.0, se_log=0.2)
    assert z > 0
    assert lo < 2.0 < hi


def test_ratio_ci_and_z_nan_se():
    z, lo, hi = _ratio_ci_and_z(1.5, se_log=np.nan)
    assert all(math.isnan(v) for v in [z, lo, hi])


def test_ratio_ci_and_z_zero_ratio():
    z, lo, hi = _ratio_ci_and_z(0.0, se_log=0.1)
    assert all(math.isnan(v) for v in [z, lo, hi])


def test_ratio_ci_width_scales_with_se():
    """A larger SE should produce a wider confidence interval."""
    _, lo1, hi1 = _ratio_ci_and_z(1.5, se_log=0.1)
    _, lo2, hi2 = _ratio_ci_and_z(1.5, se_log=0.5)
    assert (hi2 - lo2) > (hi1 - lo1)


# ---------------------------------------------------------------------------
# run_ratio_zscore() integration test
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_dsr_files(tmp_path):
    """Write minimal prev_DSR.csv and inc_DSR.csv to a temp directory."""
    conditions = ["cond_A", "cond_B"]
    groups = ["subgroup_1", "subgroup_2", "Overall"]
    dates = ["2001-01-01", "2002-01-01"]

    rng = np.random.default_rng(42)

    rows = []
    for cond in conditions:
        for group in groups:
            subgroup = "" if group == "Overall" else group
            for date in dates:
                rate = rng.uniform(100, 1000)
                var = rng.uniform(50, 500)
                rows.append({
                    "Condition": cond,
                    "Group": group,
                    "Subgroup": subgroup,
                    "Date": date,
                    "Prevalence": rate,
                    "Incidence": rate * 0.1,
                    "Lower_CI": rate * 0.8,
                    "Upper_CI": rate * 1.2,
                    "DSR_Var": var,
                })

    df = pd.DataFrame(rows)
    out_dir = str(tmp_path) + "/"
    df.to_csv(f"{out_dir}prev_DSR.csv", index=False)
    df.to_csv(f"{out_dir}inc_DSR.csv", index=False)
    return out_dir


def test_run_ratio_zscore_creates_outputs(minimal_dsr_files):
    config = {
        "dir_out": minimal_dsr_files,
        "zscore": {"alpha": 0.05, "include_groups": None},
    }
    run_ratio_zscore(config)

    out_dir = f"{minimal_dsr_files}ratio_zscore/"
    assert os.path.exists(f"{out_dir}zscore_results.csv")
    assert os.path.exists(f"{out_dir}ratios_by_year.csv")
    assert os.path.exists(f"{out_dir}ratios_overall.csv")


def test_run_ratio_zscore_result_columns(minimal_dsr_files):
    config = {
        "dir_out": minimal_dsr_files,
        "zscore": {"alpha": 0.05, "include_groups": None},
    }
    run_ratio_zscore(config)

    result = pd.read_csv(f"{minimal_dsr_files}ratio_zscore/zscore_results.csv")
    expected_cols = {"Condition", "Date", "Group", "Subgroup", "Z_Score",
                     "Ratio_CI_Lower", "Ratio_CI_Upper"}
    assert expected_cols.issubset(set(result.columns))


def test_run_ratio_zscore_no_overall_in_results(minimal_dsr_files):
    """Overall rows should be excluded from the z-score output."""
    config = {
        "dir_out": minimal_dsr_files,
        "zscore": {"alpha": 0.05, "include_groups": None},
    }
    run_ratio_zscore(config)
    result = pd.read_csv(f"{minimal_dsr_files}ratio_zscore/zscore_results.csv")
    assert "Overall" not in result["Group"].values


def test_run_ratio_zscore_include_groups_filter(minimal_dsr_files):
    """include_groups should limit output to specified groups only."""
    config = {
        "dir_out": minimal_dsr_files,
        "zscore": {"alpha": 0.05, "include_groups": ["subgroup_1", "Overall"]},
    }
    run_ratio_zscore(config)
    result = pd.read_csv(f"{minimal_dsr_files}ratio_zscore/zscore_results.csv")
    groups_in_result = set(result["Group"].unique())
    assert groups_in_result == {"subgroup_1"}  # Overall excluded from z-score output

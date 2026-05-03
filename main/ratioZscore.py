"""
ratioZscore.py
==============
Calculate subgroup-to-overall ratios, expected rates, and inequality z-scores
from the directly standardised rate (DSR) outputs of the standardisation stage.

For each condition × date × subgroup combination this module computes:
  - Ratio:     subgroup DSR / overall DSR (geometric mean used for aggregation)
  - Expected:  overall DSR × mean subgroup ratio (null hypothesis estimate)
  - SE of log(ratio): sqrt( var/rate^2 + refvar/refrate^2 )
      where var and refvar are the DSR variances for the subgroup and overall
      respectively (Dobson/Breslow-Day variance formula, same units as DSR^2).
  - Z-score:   log(ratio) / SE
  - 95% CI:    ratio × exp(±1.96 × SE)

The formula for SE is drawn from the R package `dsr` (dsrr.R):
  https://rdrr.io/cran/dsr/src/R/dsrr.R

Outputs (written to ``{dir_out}ratio_zscore/``):
  - zscore_results.csv          — per condition/date/subgroup ratios, z-scores, CIs
  - ratios_by_year.csv          — geometric mean ratio per subgroup × year
  - ratios_overall.csv          — geometric mean ratio per subgroup (all years)
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as ss


# ---------------------------------------------------------------------------
# Geometric mean helper
# ---------------------------------------------------------------------------

def _geom_mean(series: pd.Series) -> float:
    """Geometric mean of a Series, ignoring zeros and NaNs."""
    s = series.replace(0, np.nan).dropna()
    if s.empty:
        return np.nan
    return float(np.exp(np.log(s).mean()))


# ---------------------------------------------------------------------------
# Ratio calculation
# ---------------------------------------------------------------------------

def _ratios_for_condition(df_cond: pd.DataFrame, measure: str, ratio_col: str) -> pd.DataFrame:
    """Return per-subgroup ratios (subgroup / Overall) for a single condition.

    ``df_cond`` is wide-form: rows = (Group, Subgroup), columns = date strings.
    """
    overall_sel = df_cond.loc["Overall"]
    # loc on a MultiIndex returns a DataFrame when there are multiple sub-rows;
    # we need a single Series (one value per date) for the division to broadcast correctly.
    overall: pd.Series = overall_sel.iloc[0] if isinstance(overall_sel, pd.DataFrame) else overall_sel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ratios = df_cond.div(overall, axis="columns")
    return pd.concat(
        [df_cond, ratios],
        axis=1,
        keys=[measure, ratio_col],
    )


def _build_ratios(df: pd.DataFrame, measure: str, conditions: list[str]) -> pd.DataFrame:
    """Compute ratios for all conditions and return a long-format DataFrame."""
    ratio_col = f"{measure} Ratio"
    pieces = []
    for cond in conditions:
        try:
            wide = df.loc[cond]  # (Group, Subgroup) × dates
        except KeyError:
            continue
        result = _ratios_for_condition(wide, measure, ratio_col)
        result = result.stack(future_stack=True).reset_index()
        result.columns = ["Group", "Subgroup", "Date", measure, ratio_col]
        result.insert(0, "Condition", cond)
        pieces.append(result)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


# ---------------------------------------------------------------------------
# Expected rate calculation
# ---------------------------------------------------------------------------

def _expected_rates(
    df: pd.DataFrame,
    measure: str,
    mean_ratios_by_year: pd.DataFrame,
    analysis_dates: list[str],
) -> pd.DataFrame:
    """Return expected rates: overall_rate × mean_subgroup_ratio.

    ``mean_ratios_by_year`` has index = Subgroup, columns = date strings.
    """
    ratio_col = f"{measure} Ratio"
    exp_col = f"Expected {measure}"
    pieces = []
    overall_rows = df[df["Group"] == "Overall"]

    for date in analysis_dates:
        date_overall = overall_rows[overall_rows["Date"] == date].set_index("Condition")[measure]
        for subgroup in mean_ratios_by_year.index:
            if date not in mean_ratios_by_year.columns:
                continue
            ratio = mean_ratios_by_year.loc[subgroup, date]
            expected = date_overall * ratio
            chunk = expected.reset_index()
            chunk.columns = ["Condition", exp_col]
            chunk["Date"] = date
            chunk["Group"] = subgroup
            pieces.append(chunk)

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


# ---------------------------------------------------------------------------
# SE and CI using the DSR ratio variance formula
# ---------------------------------------------------------------------------

def _se_log_ratio(var: float, rate: float, refvar: float, refrate: float) -> float:
    """SE of log(rate/refrate) from Dobson variance formula.

    se = sqrt( var/rate^2 + refvar/refrate^2 )

    Returns NaN when inputs are non-positive or NaN.
    """
    if any(np.isnan(v) for v in [var, rate, refvar, refrate]):
        return np.nan
    if rate <= 0 or refrate <= 0:
        return np.nan
    return float(np.sqrt(var / rate ** 2 + refvar / refrate ** 2))


def _ratio_ci_and_z(
    ratio: float, se_log: float, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Return (z_score, ci_lower, ci_upper) for a DSR ratio.

    z      = log(ratio) / se_log
    CI     = ratio × exp(±z_alpha × se_log)
    """
    if np.isnan(se_log) or se_log <= 0 or ratio <= 0:
        return np.nan, np.nan, np.nan
    z_alpha = ss.norm.ppf(1 - alpha / 2)
    z_score = np.log(ratio) / se_log
    ci_lo = ratio * np.exp(-z_alpha * se_log)
    ci_hi = ratio * np.exp(z_alpha * se_log)
    return z_score, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ratio_zscore(config: dict) -> None:
    """Compute and save ratio z-scores from DSR outputs.

    Parameters
    ----------
    config:
        Full pipeline config dict (loaded from config.yml).
    """
    dir_out: str = config["dir_out"]
    out_subdir = f"{dir_out}ratio_zscore/"
    os.makedirs(out_subdir, exist_ok=True)

    alpha: float = config.get("zscore", {}).get("alpha", 0.05)
    include_groups: list[str] | None = config.get("zscore", {}).get("include_groups")

    # --- Load DSR files -------------------------------------------------------
    prev_path = f"{dir_out}prev_DSR.csv"
    inc_path = f"{dir_out}inc_DSR.csv"

    def _load_dsr(path: str, measure: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        # Expected columns: Condition, Group, Subgroup, Date, <measure>, Lower_CI, Upper_CI, DSR_Var
        if measure not in df.columns:
            raise ValueError(f"Column '{measure}' not found in {path}. Run `strd` first.")
        return df

    prev_df = _load_dsr(prev_path, "Prevalence")
    inc_df = _load_dsr(inc_path, "Incidence")

    # Filter to requested groups, but always retain "Overall" rows needed for ratio calculation
    if include_groups is not None:
        prev_df = prev_df[prev_df["Group"].isin(include_groups) | (prev_df["Group"] == "Overall")]
        inc_df = inc_df[inc_df["Group"].isin(include_groups) | (inc_df["Group"] == "Overall")]

    # Derive analysis dates from the data
    prev_dates = sorted(prev_df["Date"].unique().tolist())
    inc_dates = sorted(inc_df["Date"].unique().tolist())

    conditions = sorted(set(prev_df["Condition"].unique()) | set(inc_df["Condition"].unique()))

    measure_info = {
        "Prevalence": (prev_df, prev_dates),
        "Incidence": (inc_df, inc_dates),
    }

    all_results: list[pd.DataFrame] = []
    mean_ratios_yearly: dict[str, pd.DataFrame] = {}
    mean_ratios_overall: dict[str, pd.Series] = {}

    for measure, (df, dates) in measure_info.items():
        ratio_col = f"{measure} Ratio"

        # Build wide-form index: (Condition, Group, Subgroup) × Date
        wide = (
            df[["Condition", "Group", "Subgroup", "Date", measure]]
            .set_index(["Condition", "Group", "Subgroup", "Date"])[measure]
            .unstack("Date")
        )

        # Compute ratios (subgroup / Overall) for each condition
        ratios_long = _build_ratios(wide, measure, conditions)
        if ratios_long.empty:
            continue

        # Geometric mean ratio per (Subgroup, Date) — used for expected rates
        mean_by_year = (
            ratios_long[ratios_long["Group"] != "Overall"]
            .groupby(["Group", "Date"])[ratio_col]
            .agg(_geom_mean)
            .unstack("Date")
        )
        mean_overall = (
            ratios_long[ratios_long["Group"] != "Overall"]
            .groupby("Group")[ratio_col]
            .agg(_geom_mean)
        )
        mean_ratios_yearly[measure] = mean_by_year
        mean_ratios_overall[measure] = mean_overall

        # Expected rates (Overall × mean ratio)
        expected = _expected_rates(ratios_long, measure, mean_by_year, dates)
        if expected.empty:
            continue

        # Merge ratios with expected rates
        merged = pd.merge(
            ratios_long,
            expected.rename(columns={"Group": "Group"}),
            on=["Condition", "Date", "Group"],
            how="left",
        )

        # Merge in DSR_Var for subgroup and overall
        var_df = df[["Condition", "Group", "Subgroup", "Date", "DSR_Var"]].copy() \
            if "DSR_Var" in df.columns else None
        overall_var = (
            df[df["Subgroup"] == ""][["Condition", "Date", "DSR_Var"]]
            .rename(columns={"DSR_Var": "DSR_Var_Overall"})
            if var_df is not None else None
        )

        if var_df is not None:
            merged = merged.merge(
                var_df.rename(columns={"DSR_Var": "DSR_Var_Subgroup"}),
                on=["Condition", "Group", "Subgroup", "Date"],
                how="left",
            )
        if overall_var is not None:
            merged = merged.merge(overall_var, on=["Condition", "Date"], how="left")

        # Z-scores and CIs using DSR ratio variance formula
        exp_col = f"Expected {measure}"

        def _row_stats(row):
            ratio = row.get(ratio_col, np.nan)
            if var_df is not None and "DSR_Var_Subgroup" in row:
                var = row["DSR_Var_Subgroup"]
                refvar = row.get("DSR_Var_Overall", np.nan)
                rate = row.get(measure, np.nan)
                refrate = row.get(exp_col, np.nan) / ratio if (
                    not np.isnan(ratio) and ratio != 0 and not np.isnan(row.get(exp_col, np.nan))
                ) else np.nan
                se = _se_log_ratio(var, rate, refvar, refrate)
            else:
                se = np.nan
            z, ci_lo, ci_hi = _ratio_ci_and_z(ratio, se, alpha)
            return pd.Series({
                "Z_Score": z,
                "Ratio_CI_Lower": ci_lo,
                "Ratio_CI_Upper": ci_hi,
                "SE_log_ratio": se,
            })

        non_overall = merged[merged["Group"] != "Overall"].copy()
        stats = non_overall.apply(_row_stats, axis=1)
        non_overall = pd.concat([non_overall, stats], axis=1)

        all_results.append(non_overall)

    if not all_results:
        print("No results to write — check that prev_DSR.csv and inc_DSR.csv exist.")
        return

    final = pd.concat(all_results, ignore_index=True)

    # Select and order output columns
    base_cols = ["Condition", "Date", "Group", "Subgroup"]
    metric_cols = []
    for measure in ("Prevalence", "Incidence"):
        ratio_col = f"{measure} Ratio"
        exp_col = f"Expected {measure}"
        for c in [measure, ratio_col, exp_col, "Z_Score", "Ratio_CI_Lower", "Ratio_CI_Upper"]:
            if c in final.columns and c not in metric_cols:
                metric_cols.append(c)

    out_cols = [c for c in base_cols + metric_cols if c in final.columns]
    final[out_cols].to_csv(f"{out_subdir}zscore_results.csv", index=False)

    pd.concat(
        {k: v for k, v in mean_ratios_yearly.items()}, names=["Measure"]
    ).to_csv(f"{out_subdir}ratios_by_year.csv")

    pd.concat(
        {k: v.to_frame() for k, v in mean_ratios_overall.items()}, names=["Measure"]
    ).to_csv(f"{out_subdir}ratios_overall.csv")

    print(f"Z-score results written to {out_subdir}")

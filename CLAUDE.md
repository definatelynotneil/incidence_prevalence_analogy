# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python pipeline for automatically estimating disease **incidence rates** and **point prevalence** from electronic health record (EHR) data, specifically CPRD (Clinical Practice Research Datalink) data linked with HES (Hospital Episode Statistics) and IMD (Index of Multiple Deprivation).

## Environment Setup

```bash
conda env create -f env.yml
conda activate incprev_analogy
```

Key dependencies: `polars`, `pandas`, `scipy`, `statsmodels`, `plotly`, `pyarrow`, `pyyaml`, `pytest`.

## Running the Pipeline

```bash
python run.py process   # preprocess raw EHR data → dat_processed.parquet (+ optional batch files)
python run.py incprev   # calculate crude incidence/prevalence
python run.py strd      # direct age-sex standardization (DSR), outputs DSR_Var column
python run.py zscore    # ratio z-scores and CIs for subgroup inequality
python run.py censor    # small number censoring (suppresses counts ≤n, default n=10)
python run.py report    # generate HTML plots and Table 1
```

Each stage depends on the previous stage's output. `zscore` requires `strd` to have run.

## Running Tests

```bash
bash test.sh
# or directly:
pytest tests/
# single test file:
pytest tests/test_smallNumCens.py
pytest tests/test_streaming.py
```

## Configuration

All parameters are set in `config.yml`. Key sections:

- **Paths:** input/output directories and filenames
- **Study dates:** separate start dates for incidence vs. prevalence
- **Parallelism:** `n_processes`, `batch_size`
- **Memory:** `create_batch_files`, `streaming_chunk_size` (see Large Dataset section)
- **Stratification:** `DEMOGRAPHY` list (strings for single variables, lists for composite groups)
- **Standardization:** reference population file and age bins
- **Z-scores:** `alpha`, optional `include_groups` filter
- **Censoring:** threshold `n`
- **Reporting:** which outputs to produce

## Architecture

### Pipeline Stages → Modules

| Stage | Module | Key Class/Function |
|---|---|---|
| `process` | `main/preprocessing.py` | `preprocessing()` |
| `incprev` | `main/IncPrev.py` → `main/ANALOGY_SCIENTIFIC/IncPrevMethods_polars.py` | `IncPrev` class |
| `strd` | `main/ANALOGY_SCIENTIFIC/IncPrevMethods.py` | `StrdIncPrev` class |
| `zscore` | `main/ratioZscore.py` | `run_ratio_zscore()` |
| `censor` | `main/smallNumCens.py` | `small_num_censor()` |
| `report` | `main/reportResults.py` | `report_results()` |

### Core Calculation Engine (`IncPrevMethods_polars.py`)

The `IncPrev` class is the statistical heart of the pipeline. It:
- Uses **Polars lazy evaluation** (`scan_parquet`) for memory efficiency by default
- Computes incidence rates (per 100,000 person-years) and point prevalence
- Calculates **Byar's method** 95% CIs on the crude rates
- Iterates monthly or yearly study windows via `date_range()`
- Supports arbitrary demographic stratification — single variables (string) and composite groups (list of strings joined with `, `)

`runAnalysis(streaming_chunk_size=N)` dispatches to streaming methods when `N` is set; otherwise uses the standard lazy-frame path. The streaming methods (`calculate_overall_inc_prev_streaming`, `calculate_grouped_inc_prev_streaming`) iterate via PyArrow `iter_batches` and accumulate numerator/denominator totals per chunk without loading the full file.

`IncPrev.py` orchestrates condition batching using **multiprocessing** (`Pool` with "spawn" context). Column-wise batch parquet files (`dat_batch_N.parquet`) are used automatically when they exist.

### Standardization (`IncPrevMethods.py`)

`StrdIncPrev` class uses Pandas for direct age-sex standardization (Dobson's method). It now also computes and outputs `DSR_Var` (the Dobson variance of each DSR, in PER_PY² units), which is required by the z-score stage.

### Z-scores (`ratioZscore.py`)

For each condition × date × subgroup, computes:
- **Ratio:** subgroup DSR / overall DSR; geometric mean used for aggregation across conditions/years
- **Expected rate:** overall DSR × geometric mean subgroup ratio
- **SE of log(ratio):** `sqrt(var/rate² + refvar/refrate²)` — the Dobson/dsrr.R formula
- **Z-score:** `log(ratio) / SE`
- **95% CI:** `ratio × exp(±1.96 × SE)`

Outputs are written to `out/ratio_zscore/`.

### Small Number Censoring (`smallNumCens.py`)

`getCrudeMap()` extracts numerators; any numerator ≤ n is suppressed (rate and CIs set to null). Output goes to `out/Publish/`.

### Reporting (`reportResults.py`, `dataScienceWorkflows/`)

- `table1_polars` (`dataScienceWorkflows/table1.py`): demographic summary Table 1
- `Visualisation` (`dataScienceWorkflows/graphing.py`): Plotly-based interactive scatter plots with CI error bars, exported as HTML

## Data Conventions

- **Condition columns** identified by prefix `BD_` (or listed explicitly in `BD_LIST`)
- **Study entry/exit:** `INDEX_DATE`, `END_DATE` (configurable via `col_index_date`, `col_end_date`)
- **Demographics:** any columns named in `DEMOGRAPHY`; composite groups joined with `, `
- **Output CSV:** `Condition`, `Date`, `Group`, `Subgroup`, `Numerator`, `Denominator`/`PersonYears`, `Incidence`/`Prevalence`, `Lower_CI`, `Upper_CI`
- **DSR output** additionally has `DSR_Var` column

## Large Dataset Memory Optimisations

Two independent opt-in mechanisms for datasets that do not fit in RAM:

### Column-wise batching (primary fix)
Set `create_batch_files: true` in `config.yml`. During `python run.py process`, `create_batch_parquet_files()` in `preprocessing_functions.py` writes `dat_batch_0.parquet`, `dat_batch_1.parquet`, … using Polars `sink_parquet` (streaming, no full load). Each file contains only the core cohort columns + demographic columns + that batch's BD_ columns. `IncPrev.py` picks these up automatically when they exist.

### Row-wise streaming (secondary fix)
Set `streaming_chunk_size: 500000` (or any int). `processBatch` passes this to `IncPrev.runAnalysis()`, which dispatches to the streaming methods in `IncPrevMethods_polars.py`. These iterate through the parquet file via PyArrow `iter_batches`, accumulating numerator/denominator sums per chunk.

**Recommended for 500 GB+ data:**
```yaml
create_batch_files: !!bool True
streaming_chunk_size: 500000
```

## Creating a New Config File

Use this section when a user asks you to create or set up a config file for a new condition group. Work through the questions below in order — each section has hard constraints that must be satisfied before proceeding to the next.

Place the finished file in `prespecifiedConfigs/config_<condition_name>.yml` and save `config.yml` as the template reference (do not modify it for condition-specific runs).

---

### Questions to ask the user (in order)

**1. Condition identity and output location**
- What is the condition (or condition group) name? This becomes the output subdirectory, e.g. `out/adhd_autism/`.
- Is there a condition mapping CSV (`condition_map_file`) that maps human-readable names to column fragments in the data? If yes, what is its filename (relative to `dir_data`)? What are the exact "Paper Short Name" values for the conditions of interest?
- If no mapping file: what are the exact `BD_` column names in the data for these conditions?

**2. Input data format**
- Is there a single combined file, or separate Gold and Aurum files?
  - Single file: what is the filename (relative to `dir_data`)?
  - Gold/Aurum: what are the Gold filename, Aurum filename, migration mapping file name, and delimiter (usually `\t`)?
- What date format do the date columns in the raw data use? (e.g. `%Y-%m-%d`, `%d/%m/%Y`). **The pipeline stores dates as ISO `%Y-%m-%d` in the processed parquet — if the raw data is already ISO, use `%Y-%m-%d`.**
- What are the patient ID column names in the CPRD data and in HES (if HES is being linked)?

**3. IMD**
- Does the raw data already contain an IMD column (e.g. `IMD_DECILE`, `IMD_QUINTILE`), or does IMD need to be linked by practice ID from a separate map file?
  - If linking by practice ID: what are the map file name(s) (can be a list for Gold + Aurum)? The linked column will be called `IMD_pracid` and is used as `source_col` for quintile derivation.
  - If already in the data: what is the exact column name? Use that as `source_col` for quintile derivation.
- Should IMD quintiles be derived? If yes: what are the desired quintile labels (defaults: Q1 most deprived … Q5 least deprived)?

**4. Additional derived columns**
- Is an age-binary split needed (e.g. under 18 / 18 and over)?
  - If yes: what is the numeric age column name? What is the threshold (default 18)? What labels?

**5. Demographic stratification**
- Which demographic columns exist in the data that should be used for stratification? Collect the **exact column names** for each:
  - Age category (required for standardisation — must be an existing column, e.g. `AGE_CATEGORY`)
  - Sex (required for standardisation)
  - Any other standalone stratifiers (e.g. `IMD_QUINTILE`, `REGION`, `ETHNICITY`)
  - Any derived columns just created above
- Which composite cross-tabulations are needed in the crude output only (e.g. `[IMD_QUINTILE, REGION]`, `[AGE_BINARY, SEX]`)? These will not be standardised.
- Which composite groups are needed for standardisation? These must include `AGE_CATEGORY` and `SEX` plus one additional variable each (e.g. `[AGE_CATEGORY, SEX, IMD_QUINTILE]`). Each will become one row in `strd.standard_breakdowns`.

**6. Study dates**
- What are the start and end dates for the incidence study window?
- What are the start and end dates for the prevalence study window? (Prevalence typically starts 6 months after incidence start to allow a burn-in period.)

**7. Standardisation reference population**
- What is the filename of the reference population CSV (relative to `dir_data`)? Default: `"UK Age-Sex Pop Structure.csv"`.
- Should the default age bins and labels be used (`[0,16,30,40,50,60,70,80,115]` / `['0-16','17-30','31-40','41-50','51-60','61-70','71-80','81+']`), or does the user need different bins?

**8. Z-scores**
- Should z-scores be computed for all standardised groups, or a subset? If a subset: which breakdown labels (these must be keys from `strd.standard_breakdowns`, **not** column names)?

**9. Reporting**
- Should Table 1 be produced? If yes: which categorical columns and which numeric columns?
- Which groups should appear in crude plots? (Use the exact column names or `"Overall"`.)
- Which groups should appear in standardised plots? (Use the breakdown label keys from `strd.standard_breakdowns`, **not** column names.)

---

### Hard constraints to enforce when building the config

**DEMOGRAPHY ↔ strd.standard_breakdowns alignment (the most common source of errors)**

Every composite list in `incprev.DEMOGRAPHY` that is intended for standardisation must have a corresponding entry in `strd.standard_breakdowns` with the columns listed in **exactly the same order**, joined by `, ` (comma-space). The strd key is an arbitrary label, but the value string must match the DEMOGRAPHY list exactly:

```yaml
# DEMOGRAPHY list entry:
- ["AGE_CATEGORY", "SEX", "IMD_QUINTILE"]

# Corresponding strd entry (key is arbitrary, value must match):
strd:
  standard_breakdowns:
    imd_quintile: "AGE_CATEGORY, SEX, IMD_QUINTILE"   # ← must match exactly
```

Composite groups that are **crude only** (no standardisation) should appear in DEMOGRAPHY but must **not** appear in `strd.standard_breakdowns`.

**strd.standard_breakdowns must always include `"Overall"`**

The key must be spelled exactly `Overall` (capital O). Its value is the base age-sex breakdown, always `"AGE_CATEGORY, SEX"` (or the equivalent column names in the user's data).

**zscore.include_groups uses strd label keys, not column names**

`zscore.include_groups` values must match the keys in `strd.standard_breakdowns` (e.g. `"imd_quintile"`, `"region"`), not the column names themselves (not `"IMD_QUINTILE"`). The `Overall` breakdown is typically excluded from zscore since it is the reference.

**report.catgs_strd uses strd label keys, not column names**

Same rule as above — `report.catgs_strd` values are strd label keys (e.g. `"imd_quintile"`), not column names. `report.catgs_crude` values are column names (or `"Overall"`).

**age_group_labels length must equal len(age_bins) − 1**

For example, bins `[0,16,30,40,50,60,70,80,115]` (9 values) require exactly 8 labels.

**imd_map_file structure depends on link_imd**

- `link_imd: true` → `imd_map_file` must be a list of one or more CSV filenames (even for a single file). Each CSV is pracid-keyed; the linked column created is `IMD_pracid`, which becomes the `source_col` for quintile derivation.
- `link_imd: false` → set `imd_map_file: null`. The `source_col` for quintile derivation is whatever IMD column already exists in the data (e.g. `IMD_DECILE`).

**condition_map_file changes how BD_LIST is interpreted**

- When set: `BD_LIST` entries are "Paper Short Name" values from the CSV — the user must supply the **exact** strings from that file (case-insensitive matching is attempted, but exact is safer).
- When null: `BD_LIST` entries are direct column names in the data (with `BD_` prefix, e.g. `BD_ADHD`).

**date_fmt must match the raw input data format, not the processed parquet**

CPRD data is typically stored with ISO dates (`%Y-%m-%d`). Use `%Y-%m-%d` unless the user confirms their raw data uses a different format. A mismatch causes silent date parsing failures in the `incprev` stage.

**All column names are case-sensitive**

Every column name in `DEMOGRAPHY`, `strd.standard_breakdowns`, `derived_columns`, `table1_catgs`, and `catgs_crude` must match the exact capitalisation of the column in the data.

---

### Checklist before saving the config

- [ ] `date_fmt` matches the raw input file date format (`%Y-%m-%d` for standard CPRD)
- [ ] Every column in `DEMOGRAPHY` exists in the processed data (including any newly derived columns)
- [ ] Every composite DEMOGRAPHY list intended for standardisation has a matching entry in `strd.standard_breakdowns` (same columns, same order)
- [ ] `strd.standard_breakdowns` contains key `"Overall"` with value `"AGE_CATEGORY, SEX"` (or equivalent)
- [ ] `zscore.include_groups` values are strd label keys, not column names; `Overall` is absent
- [ ] `report.catgs_strd` values are strd label keys; `report.catgs_crude` values are column names / `"Overall"`
- [ ] `len(age_group_labels) == len(age_bins) - 1`
- [ ] If `link_imd: true`, `imd_map_file` is a list; if `link_imd: false`, `imd_map_file` is null
- [ ] If `condition_map_file` is set, `BD_LIST` entries are Paper Short Names from that file
- [ ] `dir_out` is a unique subdirectory for this condition group (e.g. `out/adhd_autism/`)
- [ ] Usage comments at the top of the file show the correct path (`"prespecifiedConfigs/config_<name>.yml"`)
- [ ] TODO comment for `pipeline_id` is present (not yet implemented — see existing configs)

---

## Building a Standalone Executable

```bash
# Linux/Mac (requires pyinstaller)
bash build_linux.zsh
# runs: pyinstaller run.py --noconfirm -n incprev_analogy --add-data="./config.yml:."
```

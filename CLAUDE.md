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

## Building a Standalone Executable

```bash
# Linux/Mac (requires pyinstaller)
bash build_linux.zsh
# runs: pyinstaller run.py --noconfirm -n incprev_analogy --add-data="./config.yml:."
```

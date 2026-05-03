# Known Issues & Maintenance Tracker

Use this file as a persistent issue log. When using Claude Code to fix issues,
start the session with: "Read BUGS.md and fix the next unchecked item."
Commit and check off each fix individually to keep sessions short and token-efficient.

---

## P0 — Crash on import or any invocation (FIXED)

- [x] **`main/smallNumCens.py:7`** — `from distutils.dir_util import copy_tree` crashes on Python 3.12
  (`distutils` was removed). Import was unused; line deleted.

- [x] **`main/ANALOGY_SCIENTIFIC/IncPrevMethods.py:91`** — `raise 'string'` is invalid Python 3 syntax
  (raises `TypeError` instead of the intended error). Fixed: `raise ValueError(...)`.

- [x] **`main/ANALOGY_SCIENTIFIC/IncPrevMethods.py:305`** — `raise("string")` is invalid Python 3 syntax.
  Fixed: `raise ValueError(...)`.

---

## P1 — Crash or silent data corruption under specific conditions (FIXED)

- [x] **`main/ANALOGY_SCIENTIFIC/IncPrevMethods_polars.py:65-70`** — `"BASELINE_DATE_LIST"` listed twice
  in `__slots__`. Duplicate removed.

- [x] **`main/IncPrev.py:201-204`** — `mp.Pool` not wrapped in context manager; worker crash leaves
  zombie processes on the HPC node. Fixed: `with mp.get_context("spawn").Pool(...) as pool:`.

- [x] **`run.py:37-43`** — Spurious `if __name__ == "__main__":` guard around `run_incprev()` call
  prevented invocation from wrappers/test harnesses. Guard removed.

- [x] **`main/preprocessing_functions.py:395`** — `mergeCols()` low-memory path re-scanned hardcoded
  `condMerged.parquet` instead of the actual output file (`{outFile}`). Caused reads from stale/wrong
  file when `outFile` differs from the default. Fixed: `scan_parquet(f"{path_dat}{outFile}")`.

---

## P2 — Code quality / error visibility (FIXED)

- [x] **`main/reportResults.py:54`** — Bare `except:` swallowed all graph-generation errors silently.
  Replaced with `except Exception as e: print(...)`.

---

## P2 — Remaining issues (not yet fixed)

- [ ] **`main/preprocessing_functions.py:709`** — `create_batch_parquet_files()` silently drops core
  columns (`INDEX_DATE`, `END_DATE`) if they are missing from the source schema, causing a cryptic
  downstream error. Add explicit validation:
  ```python
  missing = [c for c in core_cols if c not in available]
  if missing:
      raise ValueError(f"Core columns missing from source parquet: {missing}")
  ```

- [ ] **`main/smallNumCens.py:64,74,75`** — Inconsistent path separators: `f"{dir_out}/inc_crude.csv"`
  has a double-slash when `dir_out` already ends with `/`. Low severity (OS handles it) but worth
  normalising to `f"{dir_out}inc_crude.csv"`.

- [ ] **`main/IncPrev.py:4`** — `from itertools import repeat` is imported but never used. Remove.

- [ ] **`main/ratioZscore.py:176-180`** — No check that `DSR_Var` column exists before accessing it.
  If `strd` stage was run without the Dobson variance patch, z-score stage crashes with a KeyError.
  Add: `if "DSR_Var" not in df.columns: raise ValueError("DSR_Var missing — re-run strd stage.")`.

- [ ] **`main/preprocessing_functions.py:227`** — `process_imd()` opens IMD mapping file without first
  checking it exists, giving a confusing `FileNotFoundError`. Add `if not exists(path): raise`.

- [ ] **`main/reportResults.py:58`** — `data_prev.get_column("Condition")` used even when `incprev=="inc"`.
  Should reference `dat_.get_column("Condition")` to avoid using wrong dataframe's conditions.

---

## Token-efficient workflow notes

- Fix one item per Claude Code session; commit after each fix.
- Verify with `pytest tests/` (requires `conda activate incprev_analogy` first).
- Use `/compact` when the conversation context grows large before making edits.
- Keep prompts specific: "Fix the DSR_Var KeyError in ratioZscore.py line 176 — add a column existence
  check before accessing DSR_Var; raise ValueError with a helpful message. Run pytest."

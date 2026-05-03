import gc
import logging
from typing import Optional, Union
from datetime import date, datetime
from typing import List, Dict, Callable, Iterator
from collections import defaultdict
import numpy as np
from scipy.stats import chi2
from scipy.special import ndtri
from dateutil.relativedelta import relativedelta
import polars as pl

_log = logging.getLogger(__name__)


def _rss_mb() -> str:
    """Return current process RSS in MB as a formatted string."""
    try:
        with open("/proc/self/status") as _fh:
            for _line in _fh:
                if _line.startswith("VmRSS:"):
                    return f"{int(_line.split()[1]) / 1024:.0f} MB"
    except Exception:
        pass
    return "?"

class IncPrev():
    """
    |Manages the automated incidence rate and point prevalence calculations on an output from Dexter.
    |Normal use would initialise object, then use any of:
    |            calculate_incidence
    |            calculate_prevalence
    |            calculate_grouped_incidence
    |            calculate_grouped_prevalence

    |Required Args:
    |    STUDY_END_DATE (datetime): End date of the study. Should be same as the value entered in dexter for data extract.
    |    STUDY_START_DATE (datetime): Start date of the study. Should be same as the value entered in dexter for data extract.
    |    FILENAME (string): Name of the data file.

    |Optional Args:
    |    BASELINE_DATE_LIST (list of strings): Default: []. Defines the columns to base event dates on. Defaults to all column names beginning with "BD_".
    |    DEMOGRAPHY (list of strings): Default: []. Defines columns to use as grouping variables. Defaults to any of AGE_CATG, SEX, ETHNICITY, COUNTRY, HEALTH_AUTH, TOWNSEND found in the column names.
    |    cols (list of strings): Default: None. List of column names to pass into usecols of pd.read_csv in IncPrev.read(). Must include INDEX_DATE, END_DATE, additionally with all BASELINE_DATE_LIST and DEMOGRAPHY. Used for efficiency.
    |    skiprows_i (list of ints): Default: None. List of indexes to pass into skiprows of pd.read_csv in IncPrev.read(). Indexes must be inclusive of header (header index = 0).
    |    read_data (bool): Default True. If True, FILENAME arge must be defined. If false, DEMOGRAPHY, BASELINE_DATE_LIST args must be defined. Defines whether raw_data will be assigned during (True) or post (False) initialisation.
    |    SMALL_FP_VAL (float): Default: 1e-8. Small constant used during calculations to avoid values of 0? [check].
    |    DAYS_IN_YEAR (float): Default: 365.25. Constant defining number of days in a year.
    |    PER_PY (string): Default: 100_000. Number of person years represented by incidence and prevalence values.
    |    ALPHA (float): Default: 0.05. Significance level for calulating error using Byar's method.
    |    INCREMENT_BY_MONTH (int): Default: 12. Number of months in each incidence/prevalence calculation.

    |Attributes:
    |    PER_PY (string): Defined above.
    |    SMALL_FP_VAL (string): Defined above.
    |    DAYS_IN_YEAR (string): Defined above.
    |    ALPHA (float): Defined above.
    |    STUDY_END_DATE (string): Defined above.
    |    STUDY_START_DATE (string): Defined above.
    |    INCREMENT_BY_MONTH (int): Defined above.
    |    FILENAME (string): Defined above.
    |    BASELINE_DATE_LIST (list): Defined above.
    |    DEMOGRAPHY (list): Defined above.
    |    raw_data (pl.DataFrame): The data used to calculate incidence rate and point prevalence.

    |Methods:
    |    read:
    |    byars_lower:
    |    byars_higher:
    |    save_dataframe_inc:
    |    save_dataframe_prev:
    |    point_incidence:
    |    point_prevalence:
    |    calculate_incidence:
    |    calculate_prevalence:
    |    calculate_grouped_incidence:
    |    calculate_grouped_prevalence:

    """

    __slots__ = 'PER_PY', 'ALPHA', \
        'STUDY_END_DATE', 'STUDY_START_DATE', \
        'BASELINE_DATE_LIST', 'DEMOGRAPHY', 'FILENAME', 'raw_data', \
        "date_fmt", "verbose", "DataKeys", "StudyDesignKeys", \
        "increment_years", "increment_months", "increment_days"

    def __init__(self,
                 STUDY_END_DATE: datetime,
                 STUDY_START_DATE: datetime,
                 FILENAME: str,
                 BASELINE_DATE_LIST: list[str] = [],
                 DEMOGRAPHY: list[str] = [],
                 cols: Optional[list[str]] = None,
                 read_data = True,
                 increment_years: int = 1,
                 increment_months: int = 0,
                 increment_days: int = 0,
                 PER_PY: Union[int,float] = 100_000,
                 ALPHA: float = 0.05,
                 col_index_date: str = "INDEX_DATE",
                 col_end_date: str = "END_DATE",
                 date_fmt: str = "%Y-%m-%d",
                 verbose: bool = False) -> None:

        self.PER_PY, self.ALPHA,\
        self.STUDY_END_DATE, self.STUDY_START_DATE, \
        self.DEMOGRAPHY, self.FILENAME, \
        self.increment_years, \
        self.increment_months, self.increment_days, \
        self.BASELINE_DATE_LIST = \
        PER_PY, ALPHA, \
        STUDY_END_DATE, STUDY_START_DATE, \
        DEMOGRAPHY, FILENAME,\
        increment_years, increment_months, increment_days, \
        BASELINE_DATE_LIST

        self.date_fmt = date_fmt
        self.verbose = verbose

        self.raw_data: Optional[pl.DataFrame]
        self.STUDY_END_DATE += relativedelta(years=0, months=0, days=1)

        self.DataKeys = {
                "INDEX_DATE_COL": col_index_date,
                "END_DATE_COL": col_end_date,
                "EVENT_DATE_COL": "EVENT_DATE",
                }

        if read_data == True:
            self.read(cols,)
        else:
            self.raw_data = None

        self.StudyDesignKeys = {
                "SMALL_FP_VAL": 1e-8,
                }


    def read(self, cols: Optional[Union[str,list[str]]],) -> None:
        if cols is None:
            cols = "*"
        if self.FILENAME[-3:] == "csv":
            self.raw_data = (
                    pl.scan_csv(self.FILENAME, infer_schema_length=0,)
                    .select(pl.col(cols))
            )
        elif self.FILENAME[-7:] == "parquet":
            self.raw_data = (
                    pl.scan_parquet(self.FILENAME,)
                    .select(pl.col(cols).cast(pl.Utf8))
            )
        else:
            raise Exception("Input file type not csv or parquet.")

        self.raw_data = (
            self.raw_data
            .with_columns(
                pl.col([self.DataKeys["INDEX_DATE_COL"],
                    self.DataKeys["END_DATE_COL"],]).str.to_date(format=self.date_fmt)
            )
        )

        if len(self.BASELINE_DATE_LIST) == 0:
             self.BASELINE_DATE_LIST = [col for col in self.raw_data.columns if col.startswith('BD_')]

        self.raw_data = self.raw_data.with_columns(
            pl.col(self.BASELINE_DATE_LIST).str.to_date(format=self.date_fmt,
                                                        strict=False)
        )

        #Missing stratification vars set to "null"
        catgs = list(set([sublist if isinstance(sublist, str) else item \
                for sublist in self.DEMOGRAPHY for item in sublist]))

        self.raw_data = self.raw_data.with_columns(
                pl.col(catgs).fill_null("null")
                )


    def byars_lower(self, count: int, denominator: Union[float,int]) -> float:
        if count < 10:
            b = chi2.ppf((self.ALPHA / 2), (count * 2)) / 2
            lower_ci = b / denominator
            return lower_ci
        else:
            z = ndtri(1 - self.ALPHA / 2)
            c = 1 / (9 * count)
            b = 3 * np.sqrt(count)
            lower_o = count * ((1 - c - (z / b)) ** 3)
            lower_ci = lower_o / denominator
            return lower_ci


    def byars_higher(self, count: int, denominator: Union[float,int]) -> float:
        if count < 10:
            b = chi2.ppf(1 - (self.ALPHA / 2), 2 * count + 2) / 2
            upper_ci = b / denominator
            return upper_ci
        else:
            z = ndtri(1 - self.ALPHA / 2)
            c = 1 / (9 * (count + 1))
            b = 3 * (np.sqrt(count + 1))
            upper_o = (count + 1) * ((1 - c + (z / b)) ** 3)
            upper_ci = upper_o / denominator
            return upper_ci


    def _flat_demo_cols(self) -> List[str]:
        """Return the unique individual column names referenced in DEMOGRAPHY."""
        return list(set(
            item
            for entry in self.DEMOGRAPHY
            for item in (entry if isinstance(entry, list) else [entry])
        ))

    def _iter_chunks(self, cols: List[str], chunk_size: int) -> Iterator[pl.DataFrame]:
        """Yield Polars DataFrames of ``chunk_size`` rows from FILENAME via PyArrow.

        Date columns are parsed and demographic nulls are filled on each chunk so
        that the rule expressions can be applied directly.

        Preprocessing coalesces source-prefixed BD_MEDI: columns into clean
        BD_CONDNAME columns, so cols contains the exact parquet column names.
        """
        import pyarrow.parquet as pq

        date_cols = [
            self.DataKeys["INDEX_DATE_COL"],
            self.DataKeys["END_DATE_COL"],
        ] + list(self.BASELINE_DATE_LIST)

        catg_cols = self._flat_demo_cols()

        pf = pq.ParquetFile(self.FILENAME)
        pf_schema_names = set(pf.schema_arrow.names)
        read_cols = [c for c in cols if c in pf_schema_names]

        for batch in pf.iter_batches(columns=read_cols, batch_size=chunk_size):
            chunk = pl.from_arrow(batch)
            chunk = chunk.with_columns(
                pl.col([c for c in date_cols if c in chunk.columns]).cast(pl.Utf8)
            )
            chunk = chunk.with_columns(
                pl.col([c for c in date_cols if c in chunk.columns])
                .str.to_date(format=self.date_fmt, strict=False)
            )
            existing_catgs = [c for c in catg_cols if c in chunk.columns]
            if existing_catgs:
                chunk = chunk.with_columns(pl.col(existing_catgs).fill_null("null"))
            yield chunk

    def _apply_and_sum(
        self,
        chunk: pl.DataFrame,
        exprs: List[pl.Expr],
        dates: List[str],
    ) -> pl.DataFrame:
        """Apply metric expressions to a chunk and sum each date column."""
        return chunk.select(exprs).sum().select(
            [pl.col(d) for d in dates if d in chunk.select(exprs).columns]
        )

    def _build_rate_df(
        self,
        acc: dict,
        is_incidence: bool,
    ) -> pl.DataFrame:
        """Convert a flat accumulator dict to a rated DataFrame with CIs."""
        col_name = "Incidence" if is_incidence else "Prevalence"
        rows = [
            {
                "Condition": cond,
                "Group": group,
                "Subgroup": subgroup,
                "Date": date_str,
                "Numerator": vals[0],
                "Denominator": vals[1],
            }
            for (cond, group, subgroup, date_str), vals in acc.items()
        ]
        if not rows:
            return pl.DataFrame(schema={
                "Condition": pl.Utf8, "Group": pl.Utf8, "Subgroup": pl.Utf8,
                "Date": pl.Utf8, "Numerator": pl.Float64, "Denominator": pl.Float64,
                col_name: pl.Float64, "Lower_CI": pl.Float64, "Upper_CI": pl.Float64,
            })

        df = pl.DataFrame(rows)
        df = df.with_columns(
            ((pl.col("Numerator") / pl.col("Denominator")) * self.PER_PY).alias(col_name),
            pl.struct(["Numerator", "Denominator"])
            .map_elements(
                lambda x: self.byars_lower(x["Numerator"], x["Denominator"]) * self.PER_PY,
                return_dtype=pl.Float64,
            )
            .alias("Lower_CI"),
            pl.struct(["Numerator", "Denominator"])
            .map_elements(
                lambda x: self.byars_higher(x["Numerator"], x["Denominator"]) * self.PER_PY,
                return_dtype=pl.Float64,
            )
            .alias("Upper_CI"),
        )
        return df

    def calculate_overall_inc_prev_streaming(
        self,
        is_incidence: bool,
        chunk_size: int,
    ) -> pl.DataFrame:
        """Streaming version of calculate_overall_inc_prev.

        Iterates the source file in row chunks of ``chunk_size``, accumulating
        numerator counts and denominator sums without loading the full file.
        """
        drange = list(self.date_range(
            self.STUDY_START_DATE, self.STUDY_END_DATE,
            self.increment_years, self.increment_months, self.increment_days,
        ))
        dates = [str(d.date()) for d in (drange[:-1] if is_incidence else drange)]

        core_cols = [self.DataKeys["INDEX_DATE_COL"], self.DataKeys["END_DATE_COL"]]
        cols = core_cols + list(self.BASELINE_DATE_LIST)

        # acc[(bd_col, "Overall", "", date)] = [numerator, denominator]
        acc: dict = defaultdict(lambda: [0, 0.0])

        for chunk in self._iter_chunks(cols, chunk_size):
            for bd_col in self.BASELINE_DATE_LIST:
                chunk_bd = chunk.with_columns(
                    pl.col(bd_col).alias(self.DataKeys["EVENT_DATE_COL"])
                )
                if is_incidence:
                    num_exprs = self.incidence_numerator_rule(drange)
                    den_exprs = self.incidence_denominator_rule(drange)
                else:
                    num_exprs = self.prevalence_numerator_rule(drange)
                    den_exprs = self.prevalence_denominator_rule(drange)

                num_row = chunk_bd.select(num_exprs).sum()
                den_row = chunk_bd.select(den_exprs).sum()

                for d in dates:
                    if d in num_row.columns:
                        acc[(bd_col, "Overall", "", d)][0] += num_row[d][0]
                        acc[(bd_col, "Overall", "", d)][1] += den_row[d][0]

        return self._build_rate_df(acc, is_incidence)

    def calculate_grouped_inc_prev_streaming(
        self,
        is_incidence: bool,
        chunk_size: int,
    ) -> pl.DataFrame:
        """Streaming version of calculate_grouped_inc_prev.

        Subgroup totals are accumulated per chunk via group_by so that the full
        cartesian product of subgroups × dates is never held in memory at once.
        """
        drange = list(self.date_range(
            self.STUDY_START_DATE, self.STUDY_END_DATE,
            self.increment_years, self.increment_months, self.increment_days,
        ))
        dates = [str(d.date()) for d in (drange[:-1] if is_incidence else drange)]

        catg_cols = self._flat_demo_cols()
        core_cols = [self.DataKeys["INDEX_DATE_COL"], self.DataKeys["END_DATE_COL"]]
        cols = core_cols + list(self.BASELINE_DATE_LIST) + catg_cols

        # acc[(bd_col, group_name, subgroup_value, date)] = [numerator, denominator]
        acc: dict = defaultdict(lambda: [0, 0.0])

        for chunk in self._iter_chunks(cols, chunk_size):
            for bd_col in self.BASELINE_DATE_LIST:
                for demo in self.DEMOGRAPHY:
                    if isinstance(demo, list):
                        demo_name = ", ".join(demo)
                        chunk_demo = chunk.with_columns(
                            pl.concat_str(pl.col(demo), separator=", ").alias(demo_name)
                        )
                        demo_col = demo_name
                    else:
                        chunk_demo = chunk
                        demo_col = demo

                    chunk_bd = chunk_demo.with_columns(
                        pl.col(bd_col).alias(self.DataKeys["EVENT_DATE_COL"])
                    )

                    if is_incidence:
                        num_exprs = self.incidence_numerator_rule(drange)
                        den_exprs = self.incidence_denominator_rule(drange)
                    else:
                        num_exprs = self.prevalence_numerator_rule(drange)
                        den_exprs = self.prevalence_denominator_rule(drange)

                    # Compute per-subgroup sums for this chunk
                    num_agg = (
                        chunk_bd.with_columns(num_exprs)
                        .group_by(demo_col)
                        .agg([pl.col(d).sum() for d in dates if d in
                              chunk_bd.with_columns(num_exprs).columns])
                    )
                    den_agg = (
                        chunk_bd.with_columns(den_exprs)
                        .group_by(demo_col)
                        .agg([pl.col(d).sum() for d in dates if d in
                              chunk_bd.with_columns(den_exprs).columns])
                    )

                    for row in num_agg.iter_rows(named=True):
                        sg = row[demo_col]
                        for d in dates:
                            if d in row:
                                acc[(bd_col, demo_col, sg, d)][0] += row[d]

                    for row in den_agg.iter_rows(named=True):
                        sg = row[demo_col]
                        for d in dates:
                            if d in row:
                                acc[(bd_col, demo_col, sg, d)][1] += row[d]

        return self._build_rate_df(acc, is_incidence)

    def runAnalysis(self,
                    inc: bool = True,
                    prev: bool = True,
                    streaming_chunk_size: Optional[int] = None,
                    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        streaming = streaming_chunk_size is not None
        n_conds = len(self.BASELINE_DATE_LIST)
        n_demos = len(self.DEMOGRAPHY)
        _log.info(
            "runAnalysis: %d condition(s), %d demography group(s), streaming=%s "
            "chunk_size=%s [RSS: %s]",
            n_conds, n_demos, streaming, streaming_chunk_size, _rss_mb(),
        )

        if streaming:
            if inc:
                _log.info("runAnalysis: overall incidence (streaming) [RSS: %s]", _rss_mb())
                results_inc = self.calculate_overall_inc_prev_streaming(True, streaming_chunk_size)
            else:
                results_inc = None
            if prev:
                _log.info("runAnalysis: overall prevalence (streaming) [RSS: %s]", _rss_mb())
                results_prev = self.calculate_overall_inc_prev_streaming(False, streaming_chunk_size)
            else:
                results_prev = None
            if n_demos > 0:
                if inc:
                    _log.info("runAnalysis: grouped incidence (streaming) [RSS: %s]", _rss_mb())
                    results_inc = pl.concat(
                        [results_inc, self.calculate_grouped_inc_prev_streaming(True, streaming_chunk_size)],
                        how="vertical",
                    )
                if prev:
                    _log.info("runAnalysis: grouped prevalence (streaming) [RSS: %s]", _rss_mb())
                    results_prev = pl.concat(
                        [results_prev, self.calculate_grouped_inc_prev_streaming(False, streaming_chunk_size)],
                        how="vertical",
                    )
        else:
            if inc:
                _log.info("runAnalysis: overall incidence [RSS: %s]", _rss_mb())
                results_inc = self.calculate_overall_inc_prev(is_incidence=True)
                gc.collect()
                _log.info("runAnalysis: overall incidence done [RSS: %s]", _rss_mb())
            else:
                results_inc = None
            if prev:
                _log.info("runAnalysis: overall prevalence [RSS: %s]", _rss_mb())
                results_prev = self.calculate_overall_inc_prev(is_incidence=False)
                gc.collect()
                _log.info("runAnalysis: overall prevalence done [RSS: %s]", _rss_mb())
            else:
                results_prev = None
            if n_demos > 0:
                if inc:
                    _log.info("runAnalysis: grouped incidence [RSS: %s]", _rss_mb())
                    results_inc = pl.concat(
                        [results_inc, self.calculate_grouped_inc_prev(is_incidence=True)],
                        how="vertical",
                    )
                    gc.collect()
                    _log.info("runAnalysis: grouped incidence done [RSS: %s]", _rss_mb())
                if prev:
                    _log.info("runAnalysis: grouped prevalence [RSS: %s]", _rss_mb())
                    results_prev = pl.concat(
                        [results_prev, self.calculate_grouped_inc_prev(is_incidence=False)],
                        how="vertical",
                    )
                    gc.collect()
                    _log.info("runAnalysis: grouped prevalence done [RSS: %s]", _rss_mb())

        _log.info("runAnalysis: complete [RSS: %s]", _rss_mb())
        return tuple([results_inc, results_prev])


    def date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        increment_years: int = 0,
        increment_months: int = 12,
        increment_days: int = 0,
    ) -> datetime:
        """
        A generator function to get list of datetimes between start_date and end_date, with increments.

        Args:
          start_date (datetime): study start date as provided in the study design.
          end_date (datetime): study end date as provided in the study design.
          increment_years (int): yearly increments by between start date and end date.
          increment_months (int): monthly increments by between start date and end date.
          increment_days (int): daily increments by between start date and end date.

        Returns:
            current_period (datetime)
        """
        current_period = start_date
        delta = relativedelta(years=increment_years, months=increment_months, days=increment_days)
        while current_period <= end_date:
            yield current_period
            current_period += delta


    """
    Incidence and Prevalence Rule definitions.
    """


    def prevalence_numerator_rule(self, d_range: List[datetime]) -> List[pl.Expr]:
        """
        Function definition for rules to calculate prevalence numerator.

        Args:
            d_range(List[datetime]): a list of datetime values to calculate point prevalence at.

        Returns:
            query(List[pl.Expr])

        """
        query = []
        for d in d_range:
            query.append(
                pl.when(
                    (pl.col(self.DataKeys["INDEX_DATE_COL"]) <= d)
                    & (pl.col(self.DataKeys["END_DATE_COL"]) >= d)
                    & (pl.col(self.DataKeys["EVENT_DATE_COL"]) <= d)
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(str(d.date()))
            )
        return query


    def prevalence_denominator_rule(self, d_range: List[datetime]) -> List[pl.Expr]:
        """
        Function definition for rules to calculate prevalence denominator.

        Args:
            d_range(List[datetime]): a list of datetime values to calculate point prevalence at.

        Returns:
            query(List[pl.Expr])
        """
        query = []
        for d in d_range:
            query.append(
                pl.when((pl.col(self.DataKeys["INDEX_DATE_COL"]) <= d) & \
                        (pl.col(self.DataKeys["END_DATE_COL"]) >= d))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(str(d.date()))
            )
        return query


    def incidence_numerator_rule(self, d_range: List[datetime]) -> List[pl.Expr]:
        """
        Function definition for rules to calculate incidence numerator.

        Args:
            d_range(List[datetime]): a list of datetime values to calculate period incidence at.

        Returns:
            query(List[pl.Expr])
        """
        query = []
        for i in range(1, len(d_range)):
            query.append(
                pl.when(
                    (
                        pl.col(self.DataKeys["EVENT_DATE_COL"]).is_between(
                            d_range[i - 1], d_range[i], closed="left"
                        )
                    )
                    & (pl.col(self.DataKeys["EVENT_DATE_COL"]) > \
                            pl.col(self.DataKeys["INDEX_DATE_COL"]))
                    & (pl.col(self.DataKeys["END_DATE_COL"]) >= d_range[i - 1])
                    & (pl.col(self.DataKeys["INDEX_DATE_COL"]) < d_range[i])
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(str(d_range[i - 1].date()))
            )
        return query


    def incidence_denominator_rule(self, d_range: List[datetime]) -> List[pl.Expr]:
        """
        Function definition for rules to calculate incidence denominator.

        Args:
            d_range(List[datetime]): a list of datetime values to calculate period incidence at.

        Returns:
            query(List[pl.Expr])
        """
        query = []
        for i in range(1, len(d_range)):
            delta = d_range[i] - d_range[i - 1]
            query.append(
                pl.when(
                    (
                        (
                            (pl.col(self.DataKeys["EVENT_DATE_COL"]) >= d_range[i - 1])
                            & (pl.col(self.DataKeys["EVENT_DATE_COL"]) >\
                               pl.col(self.DataKeys["INDEX_DATE_COL"]))
                        )
                        | (pl.col(self.DataKeys["EVENT_DATE_COL"]).is_null())
                    )
                    & (pl.col(self.DataKeys["END_DATE_COL"]) >= d_range[i - 1])
                    & (pl.col(self.DataKeys["INDEX_DATE_COL"]) < d_range[i])
                )
                .then(
                    pl.min_horizontal(
                        pl.col(self.DataKeys["END_DATE_COL"]),
                        pl.col(self.DataKeys["EVENT_DATE_COL"]), d_range[i]
                    )
                    .sub(pl.max_horizontal(pl.col(self.DataKeys["INDEX_DATE_COL"]), d_range[i - 1]))
                    .dt.total_days()
                    .cast(pl.Float64)
                    / delta.days
                )
                .otherwise(pl.lit(self.StudyDesignKeys["SMALL_FP_VAL"]))
                .alias(str(d_range[i - 1].date()))
            )
        return query


    def calculate_metrics(
        self,
        melted_df: pl.LazyFrame,
        rule_fn: Callable[[List[datetime]], List[pl.Expr]],
        d_range: List[datetime],
        col_list: List[str],
        rename: Dict[str, str],
    ) -> pl.LazyFrame:
        query = rule_fn(d_range)

        melted_df = (
            melted_df.with_columns(query)
            .group_by(["Condition", "Group", "Subgroup"])
            .agg(pl.col(col_list).sum())
        )

        return melted_df.melt(id_vars=["Condition", "Group", "Subgroup"], value_vars=col_list).rename(
            rename
        )


    def filter_data_for_combination(self, data: pl.LazyFrame, condition: List[str], demography: List[str]) -> pl.LazyFrame:
        if self.verbose:
            print(demography)
            print(condition)
        # Extract relevant columns
        filtered_data = data.select(
            [self.DataKeys["INDEX_DATE_COL"], self.DataKeys["END_DATE_COL"]] +\
                    condition + demography
        )

        # Rename columns to match the desired structure
        if len(demography) > 0:
            return filtered_data.rename(
                {condition[0]: self.DataKeys["EVENT_DATE_COL"], demography[0]: "Subgroup"}
            ).with_columns(
                pl.lit(condition[0]).alias("Condition"), pl.lit(demography[0]).alias("Group")
            )
        else:
            return filtered_data.rename({condition[0]:\
            self.DataKeys["EVENT_DATE_COL"]}).with_columns(
                pl.lit(condition[0]).alias("Condition"),
                pl.lit("Overall").alias("Group"),
                pl.lit("").alias("Subgroup"),
            )


    def calculate_overall_inc_prev(
        self,
        is_incidence: bool = False,
    ) -> pl.DataFrame:
        """
        Function to calculate overall incidence or prevalence for each condition.

        Args:
            data (LazyFrame): the processed study polars LazyFrame.
            study_start_date (date): study start date as defined during study extract.
            study_end_date (date): study end date as defined during study extract.
            condition_list (List): List of conditions.
            is_incidence (bool): flag for Incidence or Prevalence study
            increment_days (int): increment period by n days.
            increment_months (int): increment period by n months.
            increment_years (int): increment period by n years.

        Returns:
            polars DataFrame
        """
        #confidence_method = ByarsConfidenceInterval()
        drange = list(
            self.date_range(
                self.STUDY_START_DATE,
                self.STUDY_END_DATE,
                self.increment_years,
                self.increment_months,
                self.increment_days
            )
        )
        if self.verbose:
            print(drange)
        col_list = [str(d.date()) for d in drange]
        if self.verbose:
            print(col_list)

        col_name = "Incidence" if is_incidence else "Prevalence"
        rename_num = {"variable": "Date", "value": "Numerator"}
        rename_den = {"variable": "Date", "value": "Denominator"}
        all_results: List[pl.DataFrame] = []

        for datecol_name in self.BASELINE_DATE_LIST:
            _log.debug(
                "overall %s: condition %s [RSS: %s]",
                col_name, datecol_name, _rss_mb(),
            )
            filtered_data = self.filter_data_for_combination(self.raw_data,
                                                             [datecol_name],
                                                             [])

            if is_incidence:
                df_num = self.calculate_metrics(
                    filtered_data, self.incidence_numerator_rule,
                    drange, col_list[:-1], rename_num,
                )
                df_den = self.calculate_metrics(
                    filtered_data, self.incidence_denominator_rule,
                    drange, col_list[:-1], rename_den,
                )
            else:
                df_num = self.calculate_metrics(
                    filtered_data, self.prevalence_numerator_rule,
                    drange, col_list, rename_num,
                )
                df_den = self.calculate_metrics(
                    filtered_data, self.prevalence_denominator_rule,
                    drange, col_list, rename_den,
                )

            # Collect each condition immediately so Polars can free the
            # intermediate wide (n_rows × n_dates) materialisations rather
            # than accumulating all lazy scan plans and collecting at once.
            num_coll = df_num.collect()
            den_coll = df_den.collect()
            del df_num, df_den

            partial = num_coll.join(den_coll, on=["Condition", "Group", "Subgroup", "Date"])
            del num_coll, den_coll

            partial = partial.with_columns(
                ((pl.col("Numerator") / pl.col("Denominator")) * self.PER_PY).alias(col_name),
                pl.struct(["Numerator", "Denominator"])
                .map_elements(
                    lambda x: self.byars_lower(x["Numerator"], x["Denominator"]) * self.PER_PY,
                    return_dtype=pl.Float64,
                )
                .alias("Lower_CI"),
                pl.struct(["Numerator", "Denominator"])
                .map_elements(
                    lambda x: self.byars_higher(x["Numerator"], x["Denominator"]) * self.PER_PY,
                    return_dtype=pl.Float64,
                )
                .alias("Upper_CI"),
            )
            all_results.append(partial)
            del partial
            _log.debug(
                "overall %s: condition %s done [RSS: %s]",
                col_name, datecol_name, _rss_mb(),
            )

        return pl.concat(all_results, how="vertical")


    def calculate_grouped_inc_prev(
        self,
        is_incidence: bool = False,
    ) -> pl.DataFrame:
        """
        Function to calculate subgroup incidence or prevalence for each condition.

        Args:
            data (LazyFrame): the processed study polars LazyFrame.
            study_start_date (date): study start date as defined during study extract.
            study_end_date (date): study end date as defined during study extract.
            condition_list (List): List of conditions.
            demography_list (List): List of demography
            is_incidence (bool): flag for Incidence or Prevalence study
            increment_days (int): increment period by n days.
            increment_months (int): increment period by n months.
            increment_years (int): increment period by n years.

        Returns:
            polars DataFrame
        """
        drange = list(
            self.date_range(
                self.STUDY_START_DATE,
                self.STUDY_END_DATE,
                self.increment_years,
                self.increment_months,
                self.increment_days
            )
        )
        if self.verbose:
            print(drange)
        col_list = [str(d.date()) for d in drange]
        if self.verbose:
            print(col_list)

        col_name = "Incidence" if is_incidence else "Prevalence"
        rename_num = {"variable": "Date", "value": "Numerator"}
        rename_den = {"variable": "Date", "value": "Denominator"}
        all_results: List[pl.DataFrame] = []

        for datecol_name in self.BASELINE_DATE_LIST:
            for demo in self.DEMOGRAPHY:
                if isinstance(demo, list):
                    demo_ = ", ".join(demo)
                    data_ = self.raw_data.with_columns(
                        pl.concat_str(pl.col(demo), separator=", ").alias(demo_)
                    )
                    filtered_data = self.filter_data_for_combination(data_,
                                                                     [datecol_name],
                                                                     [demo_])
                else:
                    filtered_data = self.filter_data_for_combination(self.raw_data,
                                                                     [datecol_name],
                                                                     [demo])

                demo_label = ", ".join(demo) if isinstance(demo, list) else demo
                _log.debug(
                    "grouped %s: condition %s demo %s [RSS: %s]",
                    col_name, datecol_name, demo_label, _rss_mb(),
                )

                if is_incidence:
                    df_num = self.calculate_metrics(
                        filtered_data, self.incidence_numerator_rule,
                        drange, col_list[:-1], rename_num,
                    )
                    df_den = self.calculate_metrics(
                        filtered_data, self.incidence_denominator_rule,
                        drange, col_list[:-1], rename_den,
                    )
                else:
                    df_num = self.calculate_metrics(
                        filtered_data, self.prevalence_numerator_rule,
                        drange, col_list, rename_num,
                    )
                    df_den = self.calculate_metrics(
                        filtered_data, self.prevalence_denominator_rule,
                        drange, col_list, rename_den,
                    )

                # Collect each condition × demo immediately to free intermediate
                # wide materialisations rather than deferring all to one big collect.
                num_coll = df_num.collect()
                den_coll = df_den.collect()
                del df_num, df_den

                partial = num_coll.join(den_coll, on=["Condition", "Group", "Subgroup", "Date"])
                del num_coll, den_coll

                partial = partial.with_columns(
                    ((pl.col("Numerator") / pl.col("Denominator")) * self.PER_PY).alias(col_name),
                    pl.struct(["Numerator", "Denominator"])
                    .map_elements(
                        lambda x: self.byars_lower(x["Numerator"], x["Denominator"]) * self.PER_PY,
                        return_dtype=pl.Float64,
                    )
                    .alias("Lower_CI"),
                    pl.struct(["Numerator", "Denominator"])
                    .map_elements(
                        lambda x: self.byars_higher(x["Numerator"], x["Denominator"]) * self.PER_PY,
                        return_dtype=pl.Float64,
                    )
                    .alias("Upper_CI"),
                )
                all_results.append(partial)
                del partial
                _log.debug(
                    "grouped %s: condition %s demo %s done [RSS: %s]",
                    col_name, datecol_name, demo_label, _rss_mb(),
                )

        return pl.concat(all_results, how="vertical")




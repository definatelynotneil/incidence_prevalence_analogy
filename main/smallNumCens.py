from os import listdir, makedirs, remove
import gc
from os.path import isdir, exists
from shutil import copy
from re import match, compile
import polars as pl
from pandas import read_excel as pd_read_excel

def getCrudeMap(filePath,) -> pl.DataFrame:
    dat_ = (
            pl.read_csv(filePath, infer_schema_length=0,)
            .with_columns(
                pl.col("Numerator").cast(pl.Int64)
                )
            .select(pl.col(["Subgroup", "Date", "Condition", "Numerator", "Group",]))
            # remove intersex (done in the strd script so needed here)
            .filter(
                ~pl.col("Subgroup").str.contains(", I,"),
                ~pl.col("Subgroup").str.ends_with(", I"),
                )
            # remove age, sex part of subgroup label
            .with_columns(
                pl.col("Subgroup").map_elements(lambda x: "'" + "', '".join(["".join([char for char in label.strip() if char not in ["'", "(", ")", '"']]) for label in x.split(",")[2:]]) + "'"),
                )
            .filter(pl.col("Subgroup")!="''")
            .filter(pl.col("Group").str.starts_with("AGE_CATEGORY, SEX"))
            .with_columns(
                pl.col("Subgroup").str.replace_all("'", "").str.replace_all(" ", ""),
                )
            .select(pl.all().exclude("Group"))
            )
    dat_ = (
            dat_
            .group_by(pl.col(["Subgroup", "Date", "Condition"])).sum()
            )
    dat_overall_ = (
            pl.read_csv(filePath, infer_schema_length=0,)
            .select(pl.col(["Group", "Date", "Condition", "Numerator",]))
            .filter(pl.col("Group")=="Overall")
            .rename({"Group": "Subgroup"})
            .with_columns(
                pl.col("Numerator").cast(pl.Int64)
                )
            )
    dat_ = pl.concat([dat_, dat_overall_])

    return dat_

def small_num_censor(
        n: int,
        strd: bool,
        dir_out: str,
        dir_cens: str = "Publish/",
        ) -> None:
    ### Make publish dir #########################################################
    if not exists(f"{dir_out}{dir_cens}"):
        makedirs(f"{dir_out}{dir_cens}")

    ### Format crude files #######################################################
    output_file_inc = f"{dir_out}{dir_cens}inc_crude.csv"
    output_file_prev = f"{dir_out}{dir_cens}prev_crude.csv"

    for path_dat_, path_output_ in ((f"{dir_out}prev_crude.csv",output_file_prev), (f"{dir_out}/inc_crude.csv",output_file_inc)):
        dat = (
                pl.read_csv(path_dat_, infer_schema_length=0,)
                .with_columns(
                    pl.col("Condition").str.replace_all("_|:|BD_MEDI", "")
                    )
                )
        dat.write_csv(path_output_)

    ### Censor small counts ######################################################
    dat_crude_inc = getCrudeMap(f"{dir_out}{dir_cens}/inc_crude.csv")
    dat_crude_prev = getCrudeMap(f"{dir_out}{dir_cens}/prev_crude.csv")

    if strd:
        ## Combine dsr files with numerators from crude (need to define values to censor)
        dat_dsr_inc = pl.read_csv(f"{dir_out}inc_DSR.csv", infer_schema_length=0,)
        dat_dsr_prev = pl.read_csv(f"{dir_out}prev_DSR.csv", infer_schema_length=0,)

        dat_dsr_inc = (
                dat_dsr_inc
                .join(
                    dat_crude_inc,
                    on=["Subgroup", "Date", "Condition"],
                    how="left",
                    )
                )
        dat_dsr_inc.write_csv(f"{dir_out}{dir_cens}/inc_DSR.csv")

        dat_dsr_prev = (
                dat_dsr_prev
                .join(
                    dat_crude_prev,
                    on=["Subgroup", "Date", "Condition"],
                    how="left",
                    )
                )
        dat_dsr_prev.write_csv(f"{dir_out}{dir_cens}/prev_DSR.csv")

        del dat_dsr_inc
        del dat_dsr_prev
        gc.collect()


    ## Setting small counts and corresponding incprev to null
    def smallCountsCens(path_dat, cols, metric=None, upperCI="Upper_CI", lowerCI="Lower_CI") -> None:
        dat = pl.read_csv(path_dat, infer_schema_length=0,)
        dat = (
                dat
                .with_columns(
                    #float for compatibility when e.g. "11.0"
                    pl.col(cols).cast(pl.Float64)
                    )
                )
        for col_ in cols:
            censor = (
                   dat
                   .with_columns(
                       pl.col(col_).fill_null(0) #will be set to null at next line; needed for compatibility with apply
                       )
                   .with_columns(
                       pl.col(col_).map_elements(lambda x: False if isinstance(x, pl.Null) or x <= n else True).alias("censor")
                       )
                   .get_column("censor")
                    )
            dat = (
                    dat
                    .with_columns(
                        dat.get_column(col_).zip_with(
                            censor,
                            pl.Series([None]*censor.shape[0]),
                            ).alias(col_)
                        )
                    )
            if metric is not None:
                for metric_ in metric:
                    dat = (
                            dat
                            .with_columns(
                                dat.get_column(metric_).zip_with(
                                    censor,
                                    pl.Series([None]*censor.shape[0]),
                                    ).alias(metric_)
                                )
                            )
            if upperCI is not None:
                dat = (
                        dat
                        .with_columns(
                            dat.get_column(upperCI).zip_with(
                                censor,
                                pl.Series([None]*censor.shape[0]),
                                ).alias(upperCI)
                            )
                        )

            if lowerCI is not None:
                dat = (
                        dat
                        .with_columns(
                            dat.get_column(lowerCI).zip_with(
                                censor,
                                pl.Series([None]*censor.shape[0]),
                                ).alias(lowerCI)
                            )
                        )

        dat.write_csv(path_dat)

    if strd:
        smallCountsCens(f"{dir_out}{dir_cens}inc_DSR.csv", ["Numerator",], metric=["Incidence"])
        pl.read_csv(f"{dir_out}{dir_cens}inc_DSR.csv", infer_schema_length=0).select(pl.all().exclude("Numerator")).write_csv(f"{dir_out}{dir_cens}inc_DSR.csv")

        smallCountsCens(f"{dir_out}{dir_cens}prev_DSR.csv", ["Numerator"], metric=["Prevalence"])
        pl.read_csv(f"{dir_out}{dir_cens}prev_DSR.csv", infer_schema_length=0).select(pl.all().exclude("Numerator")).write_csv(f"{dir_out}{dir_cens}prev_DSR.csv")

    # crude
    smallCountsCens(f"{dir_out}{dir_cens}prev_crude.csv", ["Numerator", "Denominator",], metric=["Prevalence"])
    smallCountsCens(f"{dir_out}{dir_cens}inc_crude.csv", ["Numerator", "Denominator",], metric=["Incidence"])


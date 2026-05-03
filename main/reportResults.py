import polars as pl
from shutil import move
import yaml
from main.dataScienceWorkflows.graphing import Visualisation
from main.dataScienceWorkflows.table1 import table1_polars

def report_results(
        config: dict,
        table1: bool,
        crude: bool,
        strd: bool,
        ) -> None:

    file_dat = f"{config['dir_data']}{config['incprev']['filename']}"
    graphObj = Visualisation()

    graphObj.dict_lay.x_axis_ticklabelstep = config["report"]["layout"]["ticklabelstep"]

    graphObj.dict_lay.marker_opacity = config["report"]["layout"]["marker_opacity"]
    graphObj.dict_lay.marker_size = config["report"]["layout"]["marker_size"]

    graphObj.dict_lay.line_opacity = config["report"]["layout"]["line_opacity"]
    graphObj.dict_lay.line_width = config["report"]["layout"]["line_width"]

    graphObj.dict_lay.error_y_thickness = config["report"]["layout"]["error_y_thickness"]
    graphObj.dict_lay.error_y_width = config["report"]["layout"]["error_y_width"]

    ###############################################################################
    if table1:
        ## Table1
        null_values = [
                "MISSING",
                "",
                "null",
                "none",
                ]

        tb1 = table1_polars(
                file_dat,
                config["report"]["table1_catgs"],
                config["report"]["table1_nums"],
                null_values,
                )
        tb1.write_csv(f"{config['dir_out']}table1.csv")

    ###############################################################################
    ## plot graphs
    def plot_scatters(catgs, file_prev, file_inc, label_graphs):
        for dir_ in (config['dir_out'], f"{config['dir_out']}Publish/"):
            # check files exist (e.g. where not run censoring)
            try:
                data_prev = pl.read_csv(f"{dir_}{file_prev}", infer_schema_length=0,)
                data_inc = pl.read_csv(f"{dir_}{file_inc}", infer_schema_length=0,)
            except Exception as e:
                print(f"Warning: could not read {file_prev} or {file_inc} from {dir_}: {e}")
                continue

            for dat_, incprev in [(data_prev, "prev"), (data_inc, "inc")]:
                for cond_ in data_prev.get_column("Condition").unique().to_list():
                    for group_ in catgs:
                        if incprev == "prev":
                            y_var_ = "Prevalence"
                            y_var_rename_ = "Prevalence (per 100,000 persons)"
                        else:
                            y_var_ = "Incidence"
                            y_var_rename_ = "Incidence (per 100,000 py)"

                        dat_cond_plot_ = dat_.filter(
                                    pl.col("Condition")==cond_,
                                    pl.col("Group")==group_,
                                    ).with_columns(
                                        pl.col("Date").str.to_date(format="%Y-%m-%d"),
                                        pl.col(y_var_).cast(pl.Float64),
                                        pl.col("Lower_CI").cast(pl.Float64),
                                        pl.col("Upper_CI").cast(pl.Float64),
                                        ).rename({y_var_: y_var_rename_})
                        if dat_cond_plot_.filter(pl.col(y_var_rename_).is_not_null()).shape[0] == 0:
                            continue
                        label_ = graphObj.plot_scatter(
                                data = dat_cond_plot_.to_pandas(),
                                y_var = y_var_rename_,
                                x_var = "Date",
                                c_name = "Subgroup",
                                interactive = True,
                                meta_vars = ["Subgroup", y_var_rename_, "Lower_CI", "Upper_CI",],
                                out_type = "interactive",
                                toDisk = True,
                                withLine = True,
                                is_errorY = True,
                                cols_errorY = ["Lower_CI", "Upper_CI"],
                                )
                        move(f"{label_}.html", f"{dir_}{label_}_{label_graphs}_{cond_}_{group_}.html")

    if crude:
        plot_scatters(config["report"]["catgs_crude"], "prev_crude.csv", "inc_crude.csv", "crude",)
    if strd:
        plot_scatters(config["report"]["catgs_strd"], "prev_DSR.csv", "inc_DSR.csv", "strd",)

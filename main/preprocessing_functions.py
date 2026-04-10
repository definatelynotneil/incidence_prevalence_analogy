from os import rename, remove
from csv import DictWriter, reader
from polars import scan_csv, col, concat, Series, concat_list
from polars import scan_parquet
from polars import min_horizontal as plmin_horizontal
from re import sub, split
from polars import Utf8 as plUtf8
from polars import Int64 as plInt64
from polars import Categorical as plCategorical
from polars import Date as plDate
from polars import when as plwhen
from polars import all as plall
from polars import lit as pllit
import gc
import pyarrow.dataset as ds
from pyarrow.csv import CSVWriter

def rmDup(
        A_raw:str,
        B_raw:str,
        A_ind:int=2,
        B_ind:int=0,
        map_file:str="./VisionToEmisMigrators.txt",
        map_delim:str="\t",
        low_memory:bool=True,
        wdir = "./",
        parallel=False,
        logger = None,
        outFile="dat_linked.parquet",
        ):
    """
    Link datasets A and B by PracticeID
    Links the two datasets, removing duplicate practices from A.

    Compares the practices in the practice map_file across A and
    B datasets.
    Writes a csv summary, of number of practices from the mapping present in A
    or B, and whether practices exisiting in one dataset also exist in the
    other.

    Parameters:
        A_raw (str): File A path
        B_raw (str): File B path
        A_ind (int): Index of A practiceIDs column in map_file
        B_ind (int): Index of A practiceIDs column in map_file
        map_file (str): File containing mapping information
        map_delim (str): Delimeter used in mapping file
        matching_cols (bool): Do the columns match across A and B?

    Returns:
        None
    """
    if A_raw.endswith(".csv"):
        #Get names for naming of output csv
        A_name = sub(r"^.*/(.*)\.csv$", r"\1", A_raw)
        B_name = sub(r"^.*/(.*)\.csv$", r"\1", B_raw)
        #Read in data
            #infer_schema_length=0 reads all cols as utf8 (str)
            #preventing type errors when concat
        A_raw = scan_csv(f"{wdir}{A_raw}", infer_schema_length=0, low_memory=low_memory)
        B_raw = scan_csv(f"{wdir}{B_raw}", infer_schema_length=0, low_memory=low_memory)

    elif A_raw.endswith(".parquet"):
        #Get names for naming of output csv
        A_name = sub(r"^.*/(.*)\.parquet$", r"\1", A_raw)
        B_name = sub(r"^.*/(.*)\.parquet$", r"\1", B_raw)
        #Read in data
            #infer_schema_length=0 reads all cols as utf8 (str)
            #preventing type errors when concat
        A_raw = (
                scan_parquet(f"{wdir}{A_raw}", low_memory=low_memory)
                .with_columns(
                    col("*").cast(plUtf8)
                    )
                 )
        B_raw = (
                scan_parquet(f"{wdir}{B_raw}", low_memory=low_memory)
                .with_columns(
                    col("*").cast(plUtf8)
                    )
                )

    def practiceMatches(
            dict_pracMap:dict,
            uniquePracticeID_A:list,
            uniquePracticeID_B:list
            ):
        """
        Compares the practices in the practice map_file across A and
        B datasets.

        Parameters:
            dict_pracMap (dict): Mapping of practice IDs across A and B
            uniquePracticeID_A (list): Unique practice IDs in A
            uniquePracticeID_B (list): Unique practice IDs in B

        Returns:
            (dict): Counts of duplicate practices across A and B
            (list[list]): Each element defines practice ID in A, if practice\
                exists in A, and if practice exists in B
        """
        results = {}

        #Find total count of practices in the practice map for A and B
        f_exists = lambda x,y:True if x in y else False
        results['n_pracMap_inA'] = sum([f_exists(x, list(dict_pracMap.keys()))\
                for x in uniquePracticeID_A])
        results['n_pracMap_inB'] = sum([f_exists(x, \
                list(dict_pracMap.values())) for x in uniquePracticeID_B])

        def check_dictMatch(x_A, x_B, dictionary, f_exists):
            """
            Find duplicate practices across A and B

            Parameters:
                x_A (list): Unique practice IDs in A
                x_B (list): Unique practice IDs in B
                dictionary (dict): Mapping dict where key=A_Ids, value=B_Ids
                f_exists (callable): Function to check occurance of A in B

            Returns:
                (list): List, where each element is a list of\
                    Practice ID in A, if practice exists in A, and if \
                    practice exists in B.
            """
            checks = []
            for key, value in dictionary.items():
                check = [key, False, False] #Initialise list where practice ID\
                    #is as in A
                if f_exists(key, x_A):
                    check[1] = True #Is practice in A
                if f_exists(value, x_B):
                    check[2] = True #Is practice in B
                checks.append(check)
            return(checks)

        checkMatches = check_dictMatch(uniquePracticeID_A,
                uniquePracticeID_B, dict_pracMap, f_exists)

        f_compare = lambda x,y,in_A, in_B:True if x == in_A and y == \
                in_B else False
        #Get counts of practices across A and B
        results['n_inA_inB'] = sum([f_compare(x, y, True, True) for _,x,y \
                in checkMatches]) #in A and in B
        results['n_inA_notB'] = sum([f_compare(x, y, True, False) for _,x,y \
                in checkMatches]) #in A and not in B
        results['n_notA_inB'] = sum([f_compare(x, y, False, True) for _,x,y \
                in checkMatches]) #not in A and not in B
        results['n_notA_notB'] = sum([f_compare(x, y, False, False) for \
                _,x,y in checkMatches]) #not in A and not in B

        return(results, checkMatches)

    #construct the practice mapping dictionary
    dict_pracMap = {}
    with open(map_file) as f:
        csv_reader = reader(f, delimiter=map_delim)
        for i, practice in enumerate(csv_reader):
            if i != 0: #skip header
                dict_pracMap[f"p{practice[A_ind]}"] = f"p{practice[B_ind]}"

    gc.collect()

    #Run practiceMatches()
    mapping_info, dict_mappings = practiceMatches(dict_pracMap,
            (A_raw.select(
                    col('PRACTICE_ID').unique().alias("PRACTICE_ID")
                ).collect()
                .get_column("PRACTICE_ID").to_list()
            ), #Gets list of unique IDs in A
            (B_raw.select(
                    col('PRACTICE_ID').unique().alias("PRACTICE_ID")
                ).collect()
                .get_column("PRACTICE_ID").to_list()
            ) #Gets list of unique IDs in B
            )
    #Save summary counts found in practiceMatches()
    with open(f"{wdir}LINK_STATS_{A_name}_{B_name}.csv", "w") as f:
        w = DictWriter(f, mapping_info.keys())
        w.writeheader()
        w.writerow(mapping_info)
    gc.collect()

    ##Remove PRACTICE_IDs found in B and in A, from A (duplicates)
        #i.e. keep practices in A that are in the practice mapping, but do not\
        #appear in B (remove the rest from A)
    removePractice_Ids = [x for x,a,b in dict_mappings if a == True and b == True]

    dat_dedup = concat(
                    [
                        B_raw,
                        A_raw.filter(~col("PRACTICE_ID").is_in(removePractice_Ids))
                        ],
                    how="diagonal",
                    parallel=parallel,)

    if logger is None:
        print("Saving combined data")
    else:
        logger.info("Saving combined data")

    if low_memory:
        dat_dedup.sink_parquet(f"{wdir}{outFile}")
    else:
        dat_dedup.collect().write_parquet(f"{wdir}{outFile}")


def process_imd(
    file_dat,
    path_dir = "./",
    file_map = "imd_mapping.csv",
    imd_delim=",",
    i_imd_key:int=0,
    i_imd_value=1,
    low_memory=False,
    is_parquet=True,
    logger=None,
    outFile="dat_processed.parquet",
    ):
    """

    """
    meta = ["PATIENT_ID", "PRACTICE_ID", "PRACTICE_PATIENT_ID",]
    joinCol = "PRACTICE_PATIENT_ID"
    imd_dict = dict()

    with open(f"{path_dir}{file_map}", "r") as f:
        r = reader(f, delimiter=imd_delim)
        label = next(r, None)[i_imd_key]
        label = f"IMD_{label}"
        imd_dict[label] = dict()
        for line in r:
            value = str(line[i_imd_value])
            #Ensure all values contain no special characters
            value = sub(r'[^\w]', '', value)

            if label == "IMD_pracid":
                imd_dict[label][f"p{line[i_imd_key]}"] = value
            else:
                imd_dict[label][f"{line[i_imd_key]}"] = str(line[i_imd_value])

    if is_parquet:
        q1 = (
            scan_parquet(f"{path_dir}{file_dat}", low_memory=low_memory,)
        )
    else:
        q1 = (
            scan_csv(f"{path_dir}{file_dat}", low_memory=low_memory,)
        )

    q1 = (
        q1
        .select(meta)
        .with_columns(
            col(meta).cast(plUtf8),
        )
    )

    for imd_type in imd_dict.keys():
        if imd_type == "IMD_patid":
            lab_col = "PATIENT_ID"
        elif imd_type == "IMD_pracid":
            lab_col = "PRACTICE_ID"
        else:
            if logger is None:
                print("IMD script not working")
            else:
                logger.warning("IMD script not working")
            break

        map_imd = imd_dict[imd_type]
        q1 = (
            q1
            .with_columns(
                col(lab_col).replace_strict(map_imd,
                                            default=pllit(None),
                                            return_dtype=plUtf8).alias(imd_type)
                )
            )
    q1 = (
        q1
        .select([joinCol]+list(imd_dict.keys()))
        .collect().write_parquet(f"{path_dir}dat_imd.parquet")
    )

    q1
    del q1
    gc.collect()

    file_1 = f"{path_dir}{file_dat}"
    file_2 = f"{path_dir}dat_imd.parquet"

    toAdd = scan_parquet(file_2, low_memory=low_memory,)

    if is_parquet:
        combine = scan_parquet(file_1, low_memory=low_memory,)
    else:
        combine = (
            scan_csv(file_1,
                     infer_schema_length=0,
                     low_memory=low_memory,)
        )
    combine = (
        combine
        .join(toAdd, on=joinCol, how="left") #how="left" supports .sink_parquet and should be same as outer
    )
    if low_memory:
        combine.sink_parquet(f"{path_dir}{outFile}")
    else:
        combine.collect().write_parquet(f"{path_dir}{outFile}")



def mergeCols(
        path_dat: str,
        file_dat: str,
        dict_merge: dict,
        low_memory = False,
        file_type = "parquet",
        logger=None,
        outFile="condMerged.parquet",
        date_fmt="%Y-%m-%d",
        rm_old_cols=True,
        ):
    """
    NOTE: outFile cannot be condMerged_temp.parquet or condMerged_newCol.parquet
    """

    if file_type == "parquet":
        q1 = scan_parquet(f"{path_dat}{file_dat}", low_memory=low_memory,)#, infer_schema_length=0)
    else:
        q1 = scan_csv(f"{path_dat}{file_dat}", infer_schema_length=0, low_memory=low_memory,)

    if low_memory:
        for out_col, merge_cols in dict_merge.copy().items():
            if logger is None:
                print(out_col)
            else:
                logger.info(out_col)

            if len(merge_cols) > 1:

                # ensure str cols already cast to Date not cast again due to rm_cols True
                q1_schema = q1.collect_schema()
                for v_ in merge_cols:
                    if q1_schema[v_] == plUtf8:
                        q1 = q1.with_columns(col(v_).str.strptime(plDate, date_fmt))

                q1 = (
                    q1
                    .select(
                        col(["PRACTICE_PATIENT_ID"] + merge_cols)
                        )
                    .with_columns(
                        plmin_horizontal(merge_cols).alias(out_col)
                    )
                    .select(
                        col(["PRACTICE_PATIENT_ID", out_col,])
                        )
                    .collect()
                    .write_parquet(f"{path_dat}condMerged_newCol.parquet")
                )
                q1
                if rm_old_cols:
                    q1 = (
                        q1
                        .select(
                            col("*").exclude(merge_cols)
                            )
                        .join(
                            scan_parquet(f"{path_dat}condMerged_newCol.parquet"),
                            on = "PRACTICE_PATIENT_ID",
                            how = "left",
                            )
                        .sink_parquet(f"{path_dat}condMerged_temp.parquet",)
                    )
                    q1
                else:
                    q1 = (
                        q1
                        .join(
                            scan_parquet(f"{path_dat}condMerged_newCol.parquet"),
                            on = "PRACTICE_PATIENT_ID",
                            how = "left",
                            )
                        .sink_parquet(f"{path_dat}condMerged_temp.parquet",)
                        )
                    q1
                del q1
                gc.collect()
                rename(f"{path_dat}condMerged_temp.parquet",
                       f"{path_dat}{outFile}",)
                remove(f"{path_dat}condMerged_newCol.parquet")

                q1 = scan_parquet(f"{path_dat}condMerged.parquet")#, infer_schema_length=0)
            else:
                del dict_merge[out_col] #prevents deleting of unmerged cols
                gc.collect()

    else:
        for out_col, merge_cols in dict_merge.copy().items():
            if len(merge_cols) > 1:

                # ensure str cols already cast to Date not cast again due to rm_cols True
                q1_schema = q1.collect_schema()
                for v_ in merge_cols:
                    if q1_schema[v_] == plUtf8:
                        q1 = q1.with_columns(col(v_).str.strptime(plDate, date_fmt))

                q1 = (
                    q1
                    .with_columns(
                        plmin_horizontal(merge_cols).alias(out_col)
                    )
                )
            else:
                del dict_merge[out_col] #prevents deleting of unmerged cols
                gc.collect()

        rm_cols = list(dict_merge.values())
        rm_cols = [x for sublist in rm_cols for x in sublist]
        if rm_old_cols:
            q1 = (
                q1
                .select(col("*").exclude(rm_cols))
                .collect().write_parquet(f"{path_dat}{outFile}")
            )
        else:
            q1 = (
                q1
                .collect().write_parquet(f"{path_dat}{outFile}")
            )
        q1
    return "finished merging"

def combineLevels(
        path_dat: str,
        file_dat: str,
        dict_merge: dict, #column: newName:(level1, level2, ...), ...
        file_type = "parquet",
        outFile="dat_updatedLevels.parquet",
        ):
    if file_type == "parquet":
        q1 = scan_parquet(f"{path_dat}{file_dat}")#, infer_schema_length=0)
    else:
        q1 = scan_csv(f"{path_dat}{file_dat}", infer_schema_length=0)

    for col_lab, to_combo in dict_merge.items():
        for newLabel, combo in to_combo.items():
            q1 = (
                    q1
                    .with_columns(
                        col(col_lab).cast(plUtf8)
                        )
                    .with_columns(
                        (plwhen(col(col_lab).is_in(combo))
                        .then(pllit(newLabel))
                        .otherwise(col(col_lab)))
                        .alias(col_lab)
                        )
                    )
    q1.collect().write_parquet(f"{path_dat}{outFile}")



def par_to_csv(file_noExtension):
    dataset = ds.dataset(f"{file_noExtension}.parquet", format="parquet")
    schema = dataset.schema
    writer = CSVWriter(f"{file_noExtension}.csv", schema)
    for batch in dataset.to_batches():
        writer.write_batch(batch)
    print("Finished")


def link_hes(path_dat: str,
             path_hes: str,
             dat_linkCol: str,
             hes_linkCol: str,
             path_out: str,
             low_memory: bool = False,
             ):
    """
    For date cols in parquet, read as string
    """
    if path_dat.endswith("csv"):
        dat = scan_csv(path_dat, infer_schema_length=0,)
    elif path_dat.endswith("parquet"):
        dat = scan_parquet(path_dat,).with_columns(col(plDate).cast(plUtf8))
    else:
        raise Exception("Cannot determine file type")

    if path_hes.endswith("csv"):
        dat_hes = scan_csv(path_hes, infer_schema_length=0,)
    elif path_hes.endswith("parquet"):
        dat_hes = scan_parquet(path_hes,).with_columns(col(plDate).cast(plUtf8))
    else:
        raise Exception("Cannot determine file type")

    query = (
            dat
            .join(
                dat_hes,
                left_on=dat_linkCol,
                right_on=hes_linkCol,
                how="left",
                validate='1:1',
                )
            )
    if low_memory:
        query.sink_parquet(path_out)
    else:
        query.collect().write_parquet(path_out)


def create_batch_parquet_files(
    in_path: str,
    out_dir: str,
    bd_list: list,
    batch_size: int,
    core_cols: list,
    demo_cols: list,
) -> None:
    """Write one parquet file per condition-batch, containing only the columns required
    for that batch's incidence/prevalence calculation.

    Each output file (``dat_batch_0.parquet``, ``dat_batch_1.parquet``, …) contains:
    * the core cohort columns (INDEX_DATE, END_DATE, etc.)
    * all demographic stratification columns
    * only the BD_ condition columns belonging to that batch

    This reduces per-batch memory from the full processed parquet size (potentially
    hundreds of GB) to a small fraction of it.

    Parameters
    ----------
    in_path:    Absolute path to dat_processed.parquet.
    out_dir:    Directory to write batch files into (same as dir_data, typically).
    bd_list:    Ordered list of BD_ condition columns to split across batches.
    batch_size: Number of BD_ columns per batch.
    core_cols:  Always-included columns (e.g. INDEX_DATE, END_DATE).
    demo_cols:  Demographic columns (flattened from DEMOGRAPHY config).
    """
    import polars as pl

    available = set(pl.scan_parquet(in_path).collect_schema().names())

    batches = [bd_list[i : i + batch_size] for i in range(0, len(bd_list), batch_size)]

    for batch_id, batch_bd_cols in enumerate(batches):
        wanted = core_cols + demo_cols + list(batch_bd_cols)
        keep = [c for c in wanted if c in available]

        out_path = f"{out_dir}dat_batch_{batch_id}.parquet"
        (
            pl.scan_parquet(in_path, low_memory=True)
            .select(keep)
            .sink_parquet(out_path)
        )
        print(f"  Batch {batch_id}: wrote {len(keep)} columns → {out_path}")

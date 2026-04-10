from typing import Union
import re
import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.special import ndtri
from dateutil.relativedelta import relativedelta
from math import sqrt

class StrdIncPrev():
    """
    Not currently developed for general use yet
    """
    #add attribute dict of units at each step
    def __init__(self,
            standard_breakdowns,
            col_condition: str = "Condition",
            col_category: str = "Group",
            col_group: str = "Subgroup",
            alpha: float = 0.05,
            per_py: Union[float,int] = 100_000,
            ):
        self.standard_breakdowns = standard_breakdowns

        self.condition_col = col_condition
        self.category_col = col_category
        self.group_col = col_group

        self.raw_data_prev = None
        self.raw_data_inc = None

        self.ALPHA = alpha
        self.PER_PY = per_py


    def getReference(self,
                     file="UK Age-Sex Pop Structure.csv",
                     bins = [0,16,30,40,50,60,70,80,115],
                     groups = ['0-16','17-30','31-40','41-50','51-60','61-70','71-80','81+'],
                    ):
        df_census = pd.read_csv(file)
        df_census = df_census[df_census.Age!='Total']
        df_census['Age Group'] = pd.cut(df_census.Age.replace('90+',90).astype(int),
                                         bins,
                                         right=False,
                                         labels=groups)


        df_census['Groups'] = (df_census['Age Group'].astype(str) + df_census.Sex.map({'Female':', F',
                           'Male':', M'}))

        self.ref_standardDenom = df_census[['Groups','Count']].groupby('Groups').sum()


    #%% Standardisation Functions
    def read_num_files(self,
                       condition,
                       standard_breakdown,
                       measure='Prevalence',
                       remove_intersex=True):
        """read file with numerators and denominators which can be used to create a standardised estimate"""
        def ignoreRow(x, labs):
            skip = False
            if x in labs:
                skip = True
            else:
                for lab in labs:
                    if str(x).find(f"'{lab}'") != -1:
                        skip=True
                        break
            return skip

        dtypes = {'Numerator': None,
                  measure: 'float64'}

        if measure == 'Prevalence':
            file_name_ending = "_Prev.csv"
            col_denom = "Denominator"

            dtypes[col_denom] = 'int64'
            dtypes['Numerator'] = 'int64'

        elif measure == 'Incidence':
            file_name_ending = "_Inc.csv"
            col_denom = "PersonYears"

            dtypes[col_denom] = 'float64'
            dtypes['Numerator'] = 'float64'

        else:
            raise 'Invalid measure of burden of disease'

        stand_subgroup_file = self.standard_breakdowns[standard_breakdown]

        if measure == "Prevalence":
            df = self.raw_data_prev
        elif measure == "Incidence":
            df = self.raw_data_inc

        df = df[
                (df["Group"]==stand_subgroup_file).to_numpy()\
                        & (df["Condition"]==condition).to_numpy()
                ]

        df = df[["Date", "Subgroup", measure, "Denominator",
                "Numerator", "Lower_CI", "Upper_CI", "std_group"]]

        df.columns = ['Date', self.group_col, measure, col_denom,
                      'Numerator', 'Lower', 'Upper', 'std_group']

        df = df.astype(dtypes)

        return df

    def standardise_subgroups_years(self, df: pd.DataFrame, measure):
        dict_years = dict()
        if measure == "Prevalence":
            denom_col = "Denominator"
        else:
            denom_col = "PersonYears"

        wi_map = self.ref_standardDenom.Count.to_dict()

        for i in df.Date.unique():
            df_year = df[df.Date == i].copy()
            df_year = df_year[df_year[self.group_col] != "Ireland"]

            dict_years[f"{i}_Upper_CI"], dict_years[i] = self.dobsons_ci(
                df_year, True, measure, return_DSR=True
            )
            dict_years[f"{i}_Lower_CI"] = self.dobsons_ci(df_year, False, measure)
            dict_years[f"{i}_DSR_Var"] = df_year.groupby(self.group_col).apply(
                self.calc_dsr_var_group, denom_col, wi_map
            )

        return dict_years

    def standardise_overall_years(self, df, measure):
        dict_years = dict()
        if measure == "Prevalence":
            denom_col = "Denominator"
        else:
            denom_col = "PersonYears"

        wi_map = self.ref_standardDenom.Count.to_dict()

        for i in df.Date.unique():
            df_year = df[df.Date == i].copy()
            df_year["std_group"] = df_year["std_group"].apply(
                lambda x: x.replace("(", "").replace(")", "")
            )
            dict_years[f"{i}_Upper_CI"], dict_years[i] = self.dobsons_ci(
                df_year, True, measure, False, return_DSR=True
            )
            dict_years[f"{i}_Lower_CI"] = self.dobsons_ci(df_year, False, measure, False)
            dict_years[f"{i}_DSR_Var"] = self.calc_dsr_var_group(df_year, denom_col, wi_map)

        return dict_years

    def standardise_condition_results(self,
                                      condition,
                                      measure,
                                      overall = "Overall"):
        subgroup_dict = {}

        for i in [x for x in self.standard_breakdowns.keys() if x != "Overall"]:
            print(condition)
            df_subgroups = self.read_num_files(condition,
                                               i,
                                               measure=measure)
            subgroup_dict[i] = pd.DataFrame(self.standardise_subgroups_years(df_subgroups, measure))

        df_standardised = pd.concat(subgroup_dict,axis=0, names=[self.category_col, self.group_col])

        #overall
        df_subgroups = self.read_num_files(condition, "Overall", measure=measure)
        df_standardised.loc[(overall,overall),:] = pd.Series(self.standardise_overall_years(df_subgroups, measure))

        return df_standardised

    def standardise_all_conditions(self, measure='Prevalence'):
        disease_rates = {}

        if measure == "Prevalence":
            for cond_ in self.raw_data_prev[self.condition_col].unique():
                disease_rates[cond_] = self.standardise_condition_results(cond_, measure)
        else:
            for cond_ in self.raw_data_inc[self.condition_col].unique():
                disease_rates[cond_] = self.standardise_condition_results(cond_, measure)

        df_all_rates = pd.concat(disease_rates, names=[self.condition_col,
                self.category_col,
                self.group_col])

        def reformat(dat, measure):
            dat = dat.melt(ignore_index=False)
            cols = [re.sub(r".*[:0-9:].(.*)", "\\1", x) for x in list(dat.variable)]
            if measure == "Prevalence":
                cols = ["Prevalence" if x == "" else x for x in cols]
            elif measure == "Incidence":
                cols = ["Incidence" if x == "" else x for x in cols]

            dates = [re.sub(r"(.*[:0-9:]).*", "\\1", x) for x in list(dat.variable)]

            dat.variable = cols
            dat["Date"] = dates

            dat = dat.pivot_table(
                columns="variable",
                values="value",
                index=[self.condition_col, "Date", self.category_col, self.group_col],
            )

            return dat

        df_all_rates = reformat(df_all_rates, measure)

        df_all_rates = df_all_rates.reset_index(level=["Group", "Date"])

        return df_all_rates

    def calc_dsr_var_group(self, df_group, denom_col, wi_map) -> float:
        """Return the variance of the DSR (in PER_PY^2 units) for a single subgroup.

        Uses the formula: Var(DSR) = (1/sum_wi^2) * sum(wi^2 * Oi / ni^2),
        scaled by PER_PY^2 to match the units of the reported DSR.
        """
        df_group = df_group.copy()
        df_group["std_group"] = df_group["std_group"].apply(
            lambda x: x.replace("'", "")
        )
        wi = np.vectorize(wi_map.get)(df_group["std_group"].values)
        if None in wi:
            return np.NaN

        ni = df_group[denom_col].values
        Oi = df_group.Numerator.values
        sum_wi = sum(wi)

        if (ni == 0).any():
            return np.NaN

        var_raw = (1 / (sum_wi ** 2)) * sum((wi ** 2) * Oi / (ni ** 2))
        return var_raw * (self.PER_PY ** 2)

    def calc_ci_group(self, df_group, denom_col, wi_map, upper, return_DSR=False,):
        """
        df_group: grouped dataframe; groups are specific to a group catg, with information across ref catgs

        out: upper or lower CIs across every ref-catg for a specific group
        """
        index = df_group["std_group"]
        df_group["std_group"] = df_group["std_group"].apply(lambda x, rep1, rep2: x.replace(rep1, rep2), args=tuple(["'", ""]))

        wi = np.vectorize(wi_map.get)(df_group["std_group"].values)
        if (wi==None).any(): #No standardisation group
            print("Cannot find Standardisation Group for...")
            print(df_group)
            if return_DSR:
                return np.NAN, np.NAN
            else:
                return np.NAN

        ni = df_group[denom_col].values

        Oi = df_group.Numerator.values
        #sum_wi = self.ref_standardDenom.Count.sum()
        sum_wi = sum(wi)

        if (ni==0).any() and (Oi==0).any():
            if return_DSR:
                return np.NAN, np.NAN
            else:
                return np.NAN

        DSR = (( 1/sum_wi ) * sum( (wi*Oi)/ni ) )
        DSR_var = (( 1/( sum_wi**2 )) * sum( ((wi**2)*Oi) / (ni**2) ))
        O = Oi.sum()
        O_var = O

        if upper:
            O_higher = self.byars_higher(O, sum(ni), return_limits=True)
            out = DSR + ( sqrt( (DSR_var / O_var) ) * (O_higher - O) )
        else:
            O_lower = self.byars_lower(O, sum(ni), return_limits=True)
            out = DSR + ( sqrt( (DSR_var / O_var) ) * (O_lower - O) )

        if return_DSR:
            #print(df_group)
            #print(pd.DataFrame({"grp": df_group["std_group"], "ni":ni, "Oi":Oi, "wi":wi, "sum_wi":[sum_wi]*len(wi)}))
            #breakpoint()
            return out, DSR
        else:
            return out

    def dobsons_ci(self, df_year, upper = True, measure="Prevalence", group=True, return_DSR=False):
        """
        df_year: entire dataframe for a single year
        """
        if measure == "Prevalence":
            denom_col = "Denominator"
        elif measure == "Incidence":
            denom_col = "PersonYears"
        else:
            raise("Invalid metric")

        wi_map = self.ref_standardDenom.Count.to_dict() #wi of each age-sex catg in df_year

        #remove null 'measure' values
       # df_year = df_year[(df_year[measure]).notna()]
        if group:
            out = df_year.groupby([self.group_col]).apply(self.calc_ci_group, denom_col, wi_map, upper, return_DSR,)

            if return_DSR:
                return out.apply(lambda x: x[0])*self.PER_PY, out.apply(lambda x: x[1])*self.PER_PY
            else:
                return out*self.PER_PY
        else:
            out = self.calc_ci_group(df_year, denom_col, wi_map, upper, return_DSR)
            if return_DSR:
                return out[0]*self.PER_PY, out[1]*self.PER_PY
            else:
                return out*self.PER_PY

    def byars_lower(self, count, denominator, return_limits=False): #return_limit change name return_freq
        if count < 10:
            b = chi2.ppf((self.ALPHA / 2), (count * 2)) / 2
            if return_limits:
                return b
            else:
                lower_ci = b / denominator
                return lower_ci
        else:
            z = ndtri(1 - self.ALPHA / 2)
            c = 1 / (9 * count)
            b = 3 * np.sqrt(count)
            lower_o = count * ((1 - c - (z / b)) ** 3)
            if return_limits:
                return lower_o
            else:
                lower_ci = lower_o / denominator
                return lower_ci

    def byars_higher(self, count, denominator, return_limits=False):
        if count < 10:
            b = chi2.ppf(1 - (self.ALPHA / 2), 2 * count + 2) / 2
            if return_limits:
                return b
            else:
                upper_ci = b / denominator
                return upper_ci
        else:
            z = ndtri(1 - self.ALPHA / 2)
            c = 1 / (9 * (count + 1))
            b = 3 * (np.sqrt(count + 1))
            upper_o = (count + 1) * ((1 - c + (z / b)) ** 3)
            if return_limits:
                return upper_o
            else:
                upper_ci = upper_o / denominator
                return upper_ci

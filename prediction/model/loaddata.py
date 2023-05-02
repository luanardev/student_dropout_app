import os
import pandas as pd
import numpy as np


def process(csv_file):

    if os.path.exists("processed_data.csv"):
        return "processed_data.csv"
    else:
        # load data
        df = pd.read_csv(csv_file)

        # check percentage of missing values
        nullVals = df[df.columns[df.isnull().any()]].isnull().sum() * 100 / df.shape[0]

        # drop disregarded columns
        df.drop(
            ['ODLCenter(OC)', 'Religion(RLGN)', 'DisabilityIndicator(DI)', 'Vulnarability(VNBLT)', 'IncomeSource(IS)',
             'CurrentResidence(CR)'], axis=1, inplace=True)

        # Replace age with median
        stat_Age = df['Age'].describe()
        median_Age = df['Age'].median()
        df['Age'].replace(to_replace=0, value=median_Age, inplace=True)

        # replace null values with zeros
        # dataset with filled missing values
        df.fillna(0, inplace=True)

        df = convert_data(df)

        # drop columns below correlation below threshold of 0.2
        df.drop(['District(DSRCT)', 'ProgrammeCode(PC)'],
            axis=1, inplace=True)
 
        df.to_csv('processed_data.csv', index=False)

        return "processed_data.csv"


def convert_data(df):
    cols = df.columns.values
    for col in cols:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            col_vals = df[col].values.tolist()
            unique_elements = set(col_vals)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[col] = list(map(convert_to_int, df[col]))

    return df

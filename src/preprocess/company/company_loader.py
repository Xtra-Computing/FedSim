import pandas as pd
import os
import sys

from utils import move_item_to_start_, move_item_to_end_


def load_loan(loan_path):
    print("Loading loan from {}".format(loan_path))
    loan_df = pd.read_csv(loan_path)
    loan_df.drop(columns=['Name'], inplace=True)

    loan_df.info(verbose=True)

    labels = loan_df['SBA_Appv'].to_numpy()
    loan_data = loan_df.drop(columns=['SBA_Appv']).to_numpy()

    return loan_data, labels


def load_both(loan_path, company_path, host_party='loan'):
    if host_party == 'loan':
        print("Loading loan from {}".format(loan_path))
        loan_df = pd.read_csv(loan_path)

        print("Loading company from {}".format(company_path))
        company_df = pd.read_csv(company_path)

        labels = loan_df['SBA_Appv'].to_numpy()
        loan_df.drop(columns=['SBA_Appv'], inplace=True)

        loan_cols = list(loan_df.columns)
        move_item_to_end_(loan_cols, ['Name'])
        loan_df = loan_df[loan_cols]
        print("Current loan columns {}".format(loan_df.columns))

        company_cols = list(company_df.columns)
        move_item_to_start_(company_cols, ['name'])
        company_df = company_df[company_cols]
        print("Current company columns {}".format(company_df.columns))

        data1 = loan_df.to_numpy()
        data2 = company_df.to_numpy()
    else:
        assert False

    return [data1, data2], labels


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/company")  # change working directory
    company_df = pd.read_csv("company_clean.csv")
    loan_df = pd.read_csv("loan_clean.csv")

    merge_df = company_df.merge(loan_df, how='inner', on='title')


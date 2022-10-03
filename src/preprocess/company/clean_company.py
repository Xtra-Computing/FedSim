import pandas as pd
import os
import sys
import re


def clean_company(company_path, out_path):
    company_df = pd.read_csv("company.csv")
    us_company_df = company_df[company_df['country'] == 'united states']
    us_company_df = us_company_df.applymap(lambda s: s.lower() if type(s) == str else s)
    us_company_df.drop(columns=['Unnamed: 0', 'domain', 'year founded', ''])
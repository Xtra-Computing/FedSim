import os
import sys
import pickle
import re

import pandas as pd


def clean_fma(fma_path, out_clean_fma_path):
    fma_titles = []
    fma_data = []
    fma_labels = []
    print("Reformatting fma data")
    with open(fma_path, 'rb') as f:
        fma_data_labels = pickle.load(f)
        for title, datum, label in fma_data_labels:
            title = re.sub(r'\W', '', title)
            if len(title) > 0:
                fma_titles.append(title.lower())
                fma_data.append(datum)
                fma_labels.append(label)
    fma_df = pd.DataFrame(fma_data)
    fma_df['title'] = fma_titles
    fma_df['label'] = fma_labels

    # remove duplicate titles
    fma_df.set_index('title', inplace=True)
    fma_df = fma_df[~fma_df.index.duplicated(keep="first")]

    print("Saving to {}".format(out_clean_fma_path))
    fma_df.to_csv(out_clean_fma_path)
    print("Done")


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/song")  # change working directory
    clean_fma("fma.pkl", "fma_clean.csv")

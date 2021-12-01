import numpy as np
import pandas as pd
import os
import wget
import zipfile


def load_frog(path, download=True, label_col='Species'):
    assert label_col in ['Family', 'Genus', 'Species'], "Undefined label column {}".format(label_col)

    if download and not os.path.isfile(path):
        print("Downloading frog dataset")
        wget.download("https://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran%20Calls%20(MFCCs).zip",
                      out="data/frog.zip")
        with zipfile.ZipFile("data/frog.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")

    data_labels_df = pd.read_csv(path, usecols=range(0, 25))
    data_df = data_labels_df.iloc[:, :22]
    labels_df = data_labels_df[label_col].astype('category').cat.codes

    data = data_df.to_numpy()
    labels = labels_df.to_numpy()

    return data, labels
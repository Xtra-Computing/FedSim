import pandas as pd
import os
import sys

from utils import move_item_to_start_, move_item_to_end_


def load_msd(msd_path):
    print("Loading MSD from {}".format(msd_path))
    msd_df = pd.read_csv(msd_path)
    msd_df.drop(columns=['track_id', 'title', 'artist'], inplace=True)

    # msd_df = msd_df[msd_df['year'] > 1960]
    msd_df.info(verbose=True)

    labels = msd_df['year'].to_numpy()
    msd_data = msd_df.drop(columns=['year']).to_numpy()

    return msd_data, labels


def load_fma(fma_path):
    print("Loading FMA from {}".format(fma_path))
    fma_df = pd.read_csv(fma_path)
    fma_df.drop(columns=['title'], inplace=True)

    fma_df.info(verbose=True)

    labels = fma_df['label'].to_numpy()
    fma_data = fma_df.drop(columns=['label']).to_numpy()

    return fma_data, labels


def load_both(msd_path, fma_path, host_party='msd'):
    if host_party == 'msd':
        print("Loading MSD from {}".format(msd_path))
        msd_df = pd.read_csv(msd_path)

        print("Loading FMA from {}".format(fma_path))
        fma_df = pd.read_csv(fma_path)

        msd_df.drop(columns=['track_id', 'artist'], inplace=True)

        labels = msd_df['year'].to_numpy()
        msd_df.drop(columns=['year'], inplace=True)
        fma_df.drop(columns=['label'], inplace=True)

        msd_cols = list(msd_df.columns)
        move_item_to_end_(msd_cols, ['title'])
        msd_df = msd_df[msd_cols]
        print("Current MSD columns {}".format(msd_df.columns))

        fma_cols = list(fma_df.columns)
        move_item_to_start_(fma_cols, ['title'])
        fma_df = fma_df[fma_cols]
        print("Current FMA columns {}".format(fma_df.columns))

        data1 = msd_df.to_numpy()
        data2 = fma_df.to_numpy()
    elif host_party == 'fma':
        print("Loading MSD from {}".format(msd_path))
        msd_df = pd.read_csv(msd_path)

        print("Loading FMA from {}".format(fma_path))
        fma_df = pd.read_csv(fma_path)

        msd_df.drop(columns=['track_id', 'artist', 'year'], inplace=True)
        labels = fma_df['label'].to_numpy()
        fma_df.drop(columns=['label'], inplace=True)

        msd_cols = list(msd_df.columns)
        move_item_to_start_(msd_cols, ['title'])
        msd_df = msd_df[msd_cols]
        print("Current MSD columns {}".format(msd_df.columns))

        fma_cols = list(fma_df.columns)
        move_item_to_end_(fma_cols, ['title'])
        fma_df = fma_df[fma_cols]
        print("Current FMA columns {}".format(fma_df.columns))

        data1 = fma_df.to_numpy()
        data2 = msd_df.to_numpy()
    else:
        assert False

    return [data1, data2], labels


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/song")  # change working directory
    fma_df = pd.read_csv("fma_clean.csv")
    msd_df = pd.read_csv("msd_clean.csv")

    merge_df = fma_df.merge(msd_df, how='inner', on='title')


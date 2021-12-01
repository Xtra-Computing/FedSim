import os
import sys
import pickle
import re
from pathlib import Path

import pandas as pd
import numpy as np

from tqdm import tqdm
import h5py
import deprecation


def generate_msd(million_song_dir, tracks_per_year_path, out_msd_path):
    num_songs = 0
    for _ in Path(million_song_dir).rglob('*.h5'):
        num_songs += 1
    print("There are {} songs in the dataset".format(num_songs))

    id_to_title_artist = {}
    with open(tracks_per_year_path, 'r') as f:
        for i, row in enumerate(f):
            year, track_id, artist, title = row.split('<SEP>')
            assert track_id not in id_to_title_artist   # id should not be duplicated
            id_to_title_artist[track_id] = [title, artist, year]

    timbre_vec_all = np.zeros([0, 90])      # 12 average features and 78 covariance features
    track_info_all = []
    for path in tqdm(Path(million_song_dir).rglob('*.h5'), total=num_songs):
        with h5py.File(path, 'r') as f:
            # read track id
            assert len(f['analysis']['songs']) == 1, "More than 1 song in a h5 file"
            track_id = f['analysis']['songs'][0]['track_id'].decode('utf-8')

            if track_id in id_to_title_artist:
                # get title, artist and year from id_to_title_artist
                track_info = id_to_title_artist[track_id]
                track_info_all.append([track_id] + track_info)

                # read timbre information similarly as YearPredictMSD.txt in UCI
                timbre_matrix = f['analysis']['segments_timbre']
                timbre_avg = np.average(timbre_matrix, axis=0)  # 12 timbre averages
                timbre_cov_matrix = np.cov(timbre_matrix, rowvar=False)  # 12 x 12 covariance matrix
                timbre_cov = timbre_cov_matrix[np.triu_indices(timbre_cov_matrix.shape[0])]  # flatten upper triangle
                timbre_vec = np.concatenate([timbre_avg, timbre_cov]).reshape(1, -1)
                timbre_vec_all = np.concatenate([timbre_vec_all, timbre_vec], axis=0)

    print("Finished. Got {} tracks".format(len(track_info_all)))

    msd_df = pd.DataFrame(track_info_all, columns=['track_id', 'title', 'artist', 'year'])
    msd_df = pd.concat([msd_df, pd.DataFrame(timbre_vec_all)], axis=1)

    print("Saving to file")
    msd_df.to_csv(out_msd_path, index=False)
    print("Saved to {}".format(out_msd_path))


@deprecation.deprecated()
def __clean_msd(msd_path, out_clean_msd_path):
    msd_titles = []
    msd_data = []
    msd_labels = []
    print("Reformatting msd data")
    with open(msd_path, 'rb') as f:
        msd_data_labels = pickle.load(f)
        for title, datum, label in msd_data_labels:
            # title = "".join(title.split())  # remove all whitespaces
            title = re.sub(r'\W', '', title)
            if len(title) > 0:
                msd_titles.append(title.lower())
                msd_data.append(datum)
                msd_labels.append(label)
    msd_df = pd.DataFrame(msd_data)
    msd_df['title'] = msd_titles
    msd_df['label'] = msd_labels

    # remove duplicate titles
    msd_df.set_index('title', inplace=True)
    msd_df = msd_df[~msd_df.index.duplicated(keep="first")]

    # filter out extreme years
    msd_df = msd_df[msd_df['label'] > 1970]

    print("Saving to {}".format(out_clean_msd_path))
    msd_df.to_csv(out_clean_msd_path)
    print("Done")


def clean_msd(msd_path, out_clean_msd_path):
    print("Loading from {}".format(msd_path))
    msd_df = pd.read_csv(msd_path)
    print("Loaded {} tracks".format(len(msd_df.index)))

    print("Encode titles")
    msd_df['title'] = msd_df['title'].apply(lambda s: re.sub(r'\W', '', s).lower())
    msd_df = msd_df[msd_df['title'].str.len() > 0]
    print("Done")

    print("Removing duplicated titles")
    # remove duplicate titles
    msd_df.set_index('title', inplace=True)
    msd_df = msd_df[~msd_df.index.duplicated(keep="first")]
    print("Done. Got {} tracks".format(len(msd_df.index)))

    print("Saving to {}".format(out_clean_msd_path))
    msd_df.to_csv(out_clean_msd_path)
    print("Done")


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/song")  # change working directory

    generate_msd("MillionSong", "tracks_per_year.txt", "msd_full.csv")
    clean_msd("msd_full.csv", "msd_clean.csv")

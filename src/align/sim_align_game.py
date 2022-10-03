import os
import sys
import csv
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


def sim_align_game(steam_data_path, ign_data_path, sim_score_path, out_sim_aligned_path, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    ign_map = {}
    cnt = 0
    with open(ign_data_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            game = row[0]
            assert game not in ign_map
            ign_map[game] = cnt
            cnt += 1
        csvfile.close()

    steam_map = {}
    cnt = 0
    with open(steam_data_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            game = row[0]
            if game not in steam_map:
                steam_map[game] = int(row[1])
            # else:
            #     assert steam_map[game] == int(row[1])
            cnt += 1
        csvfile.close()

    with open(sim_score_path, 'rb') as f:
        sim_scores = pickle.load(f)

    # align on titles
    align_info = []
    num_pairs = len(sim_scores)
    print("Start aligning, got {} pairs".format(num_pairs))
    for (s_title, i_title), score in sim_scores.items():
        if s_title in steam_map:
            align_info.append([steam_map[s_title], ign_map[i_title], score])
        else:
            num_pairs -= 1
    align_info_df = pd.DataFrame(align_info, columns=['appid', 'ign_index', 'sim_score'])
    print("Aligning finished. {} pairs remained.".format(num_pairs))

    # merge ign records
    print("Merging ign records")
    ign_game = pd.read_csv(ign_data_path)
    ign_game['ign_index'] = range(ign_game.shape[0])
    ign_align = pd.merge(align_info_df, ign_game, on='ign_index', sort=False)
    ign_align = ign_align.drop(columns=['ign_index'])
    ign_align.rename({'title': 'ign_title'}, axis=1, inplace=True)
    print("Finished merging with ign records, got {} lines".format(len(ign_align.index)))

    # merge steam records
    print("Merging steam records")
    steam_data = pd.read_csv(steam_data_path)
    steam_data.rename({'title': 'steam_title'}, axis=1, inplace=True)
    two_party_data_df = pd.merge(steam_data, ign_align, how='left', on='appid', sort=True)
    print("Finished merging with steam records, got {} lines".format(len(two_party_data_df.index)))

    # save aligned data to file
    print("Saving aligned data to file")
    two_party_data_df.to_csv(out_sim_aligned_path, index=False)
    print("Saved")


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../")  # change working directory
    root = "data/"
    steam_data_train_path = root + "steam_data_train.csv"
    steam_data_val_path = root + "steam_data_val.csv"
    steam_data_test_path = root + "steam_data_test.csv"
    ign_data_path = root + "ign_game_clean.csv"
    sim_score_train_path = root + "sim_score_game_train.csv"
    sim_score_val_path = root + "sim_score_game_val.csv"
    sim_score_test_path = root + "sim_score_game_test.csv"
    out_sim_aligned_train_path = root + "sim_aligned_game_train.csv"
    out_sim_aligned_val_path = root + "sim_aligned_game_val.csv"
    out_sim_aligned_test_path = root + "sim_aligned_game_test.csv"
    print("Align training set")
    sim_align_game(steam_data_train_path, ign_data_path, sim_score_train_path, out_sim_aligned_train_path)
    print("Align validation set")
    sim_align_game(steam_data_val_path, ign_data_path, sim_score_val_path, out_sim_aligned_val_path)
    print("Align test set")
    sim_align_game(steam_data_test_path, ign_data_path, sim_score_test_path, out_sim_aligned_test_path)
    print("Done.")

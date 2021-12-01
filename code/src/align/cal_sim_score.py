import os
import sys
import random
import numpy as np
import pandas as pd
from nltk.metrics.distance import edit_distance
import pickle
from tqdm import tqdm


def similarity_score(a: str, b: str):
    return 1 - edit_distance(a, b) / max(len(a), len(b))


def cal_sim_score(steam_data_path, ign_data_path, out_sim_score_path, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # load raw data from steam and ign
    steam_data_df = pd.read_csv(steam_data_path)
    ign_data_df = pd.read_csv(ign_data_path)

    steam_title_df = steam_data_df['title']
    ign_title_df = ign_data_df['title']

    # extract data
    data_steam = steam_title_df.to_numpy()
    data_ign = ign_title_df.to_numpy()
    data_steam = np.unique(data_steam)
    data_ign = np.unique(data_ign)
    print("Unique steam titles: {}, unique ign titles: {}".format(len(data_steam), len(data_ign)))

    # blocking
    block_dict_steam = {}
    for record in data_steam:
        key = record[:3]
        if key in block_dict_steam:
            block_dict_steam[key].append(record)
        else:
            block_dict_steam[key] = [record]

    block_dict_ign = {}
    for record in data_ign:
        key = record[:3]
        if key in block_dict_ign:
            block_dict_ign[key].append(record)
        else:
            block_dict_ign[key] = [record]

    print("#blocks in steam: {}".format(len(block_dict_steam)))
    print("#blocks in ign: {}".format(len(block_dict_ign)))

    # Compare
    title_sim_scores = {}
    for s_key, s_block in tqdm(block_dict_steam.items()):
        if s_key not in block_dict_ign:
            continue

        i_block = block_dict_ign[s_key]
        for s_title in s_block:
            for i_title in i_block:
                sim_score = similarity_score(s_title, i_title)
                title_sim_scores[(s_title, i_title)] = sim_score

        # print("Block {} matched".format(s_key))
    print("Got {} pairs".format(len(title_sim_scores)))

    # Save similarity scores
    print("Saving to {}".format(out_sim_score_path))
    with open(out_sim_score_path, 'wb') as f:
        pickle.dump(title_sim_scores, f)
    print("Saved")


if __name__ == '__main__':
    print("Calculate similarity scores for train, val & test")
    os.chdir(sys.path[0] + "/../../")  # change working directory
    root = "data/"
    steam_data_train_path = root + "steam_data_train.csv"
    steam_data_val_path = root + "steam_data_val.csv"
    steam_data_test_path = root + "steam_data_test.csv"
    ign_data_path = root + "ign_game_clean.csv"
    out_sim_aligned_train_path = root + "sim_score_game_train.csv"
    out_sim_aligned_val_path = root + "sim_score_game_val.csv"
    out_sim_aligned_test_path = root + "sim_score_game_test.csv"
    cal_sim_score(steam_data_train_path, ign_data_path, out_sim_aligned_train_path)
    cal_sim_score(steam_data_val_path, ign_data_path, out_sim_aligned_val_path)
    cal_sim_score(steam_data_test_path, ign_data_path, out_sim_aligned_test_path)
    print("Done.")

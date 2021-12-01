import os
import sys
import csv
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_title(title: str):
    game = title.lower().replace(' ', '')
    game = ''.join(filter(str.isalnum, game))
    return game


def exact_align_game(steam_data_path, ign_data_path, out_exact_aligned_path, save_unmatched=False, seed=0):
    """
    Exact align 'game' dataset on column 'title'
    :param steam_data_path: path of pre-processed steam data (sampled and negative sampled)
    :param ign_data_path: Cleaned IGN data
    :param out_exact_aligned_path: Output path for the aligned data
    :param save_unmatched: Whether to save unmatched steam records. If True, unknown features will be N/A.
    :param seed: Random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # load ign titles
    ign_title = set()
    ign_map = {}
    cnt = 0
    with open(ign_data_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            game = clean_title(row[0])
            ign_title.add(game)
            ign_map[game] = cnt
            cnt += 1
    print('ign title num: ', len(ign_title))

    # load steam titles
    steam_title = set()
    steam_map = {}
    cnt = 0
    with open(steam_data_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            game = clean_title(row[0])
            steam_title.add(game)
            steam_map[game] = [row[1], cnt]
            cnt += 1
    print('steam title num: ', len(steam_title))

    # match titles exactly
    align_title = ign_title.intersection(steam_title)
    align_title = list(align_title)
    print('align title num', len(align_title))

    # find index of records to be aligned
    align_info = []
    for a in align_title:
        align_info.append([int(steam_map[a][0]), steam_map[a][1], ign_map[a]])
    align_info_df = pd.DataFrame(align_info, columns=['appid', 'steam_index', 'ign_index'])

    # align records from ign
    ign_game = pd.read_csv(ign_data_path)
    ign_game['ign_index'] = range(ign_game.shape[0])
    ign_align = pd.merge(align_info_df, ign_game, on='ign_index', sort=False)
    print(ign_align)
    ign_align = ign_align.drop(columns=['title', 'steam_index', 'ign_index'])

    # align records from steam
    steam_data = pd.read_csv(steam_data_path)
    if save_unmatched:
        aligned_data_df = pd.merge(steam_data, ign_align, on='appid', how='left', sort=True)
    else:
        aligned_data_df = pd.merge(steam_data, ign_align, on='appid', sort=True)

    aligned_data_df.to_csv(out_exact_aligned_path, index=False)


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../")     # change working directory
    root = "data/"
    steam_data_train_path = root + "steam_data_train.csv"
    steam_data_val_path = root + "steam_data_val.csv"
    steam_data_test_path = root + "steam_data_test.csv"
    ign_data_path = root + "ign_game_clean.csv"
    out_exact_aligned_train_path = root + "exact_aligned_game_unmatch_train.csv"
    out_exact_aligned_val_path = root + "exact_aligned_game_unmatch_val.csv"
    out_exact_aligned_test_path = root + "exact_aligned_game_unmatch_test.csv"
    exact_align_game(steam_data_train_path, ign_data_path, out_exact_aligned_train_path, save_unmatched=True)
    exact_align_game(steam_data_val_path, ign_data_path, out_exact_aligned_val_path, save_unmatched=True)
    exact_align_game(steam_data_test_path, ign_data_path, out_exact_aligned_test_path, save_unmatched=True)
    print("Done.")

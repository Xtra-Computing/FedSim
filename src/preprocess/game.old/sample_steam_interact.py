import os
import sys
import random
import pandas as pd
import numpy as np


def sample_steam_interact(interact_path, game_path, out_interact_sample_path, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    interact_df = pd.read_csv(interact_path, header=None)
    interact_df.columns = ['steamid', 'appid']
    print(interact_df.shape)
    # print(interact_df['appid'].value_counts())
    # print(interact_df['steamid'].value_counts())
    # print(interact_df.duplicated().any())

    steam = pd.read_csv(game_path)
    app_id_df = steam[['appid']]
    print(app_id_df.shape)

    merged = pd.merge(app_id_df, interact_df, on='appid', sort=False)
    print(merged.shape)
    # print(merged['appid'].value_counts())
    # print(merged['steamid'].value_counts())
    # print(merged.duplicated().any())

    merged_sub = merged.sample(n=23000000, random_state=seed)
    print(merged_sub.shape)
    print(merged_sub['appid'].value_counts())
    print(merged_sub['steamid'].value_counts())
    print(merged_sub.duplicated().any())

    cnts = merged_sub['steamid'].value_counts()
    cnts_df = pd.DataFrame({'steamid': cnts.index, 'cnts': cnts.values})
    print(cnts_df)

    filters = cnts_df[cnts_df['cnts'] >= 20]
    print(filters)

    filters_id_df = filters[['steamid']]
    print(filters_id_df)

    sample = pd.merge(filters_id_df, merged_sub, on='steamid', sort=False)
    print(sample.shape)
    print(sample['appid'].value_counts())
    print(sample['steamid'].value_counts())
    print(sample.duplicated().any())

    sample.to_csv(out_interact_sample_path, index=False)


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../")     # change working directory
    root = "data/"
    steam_interact_path = root + "steam_interact.csv"
    steam_game_path = root + "steam_game_clean.csv"
    out_steam_interact_sample_path = root + "steam_interact_sample.csv"
    sample_steam_interact(steam_interact_path, steam_game_path, out_steam_interact_sample_path)
    print("Done.")
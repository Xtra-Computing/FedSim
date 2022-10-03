import os
import sys
import random
import numpy as np
import pandas as pd


def negative_sample_steam_interact(interact_sample_path, game_path, out_data_path, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    interact_df = pd.read_csv(interact_sample_path)
    print(interact_df.shape)

    steam_game = pd.read_csv(game_path)
    # app_id = steam_game['appid']
    # print(len(app_id))
    app_id = interact_df['appid'].unique().tolist()
    print(len(app_id))

    grouped = interact_df.groupby('steamid')
    negative = []
    partial_cnt = 0
    ratio = 1
    cnt = 0
    print('start')
    for index, value in grouped:
        exist_id = set(value['appid'])
        length = len(exist_id)

        if length * (ratio + 1) >= len(app_id):
            for a in app_id:
                if a not in exist_id:
                    negative.append([index, a])
                    partial_cnt += 1
        else:
            while len(exist_id) != length * (ratio + 1):
                new_id = app_id[random.randint(0, len(app_id) - 1)]
                if new_id not in exist_id:
                    exist_id.add(new_id)
                    negative.append([index, new_id])
        #     if cnt >= 5:
        #         break
        cnt += 1
        if cnt % 1000 == 0:
            print('current step', cnt)

    print('group cnt', cnt)
    print('partial cnt', partial_cnt)
    print('negative sample cnt', len(negative))

    negative_df = pd.DataFrame(negative, columns=['steamid', 'appid'])
    negative_df['label'] = list(np.zeros(negative_df.shape[0], dtype=np.int))
    print(negative_df.shape)

    interact_df['label'] = list(np.ones(interact_df.shape[0], dtype=np.int))
    print(interact_df.shape)

    all_interact_df = interact_df.append(negative_df)
    print(all_interact_df.shape)
    print(all_interact_df['appid'].value_counts())
    print(all_interact_df['steamid'].value_counts())
    print(all_interact_df.duplicated().any())

    all_data_df = pd.merge(steam_game, all_interact_df, on='appid', sort=False)
    print(all_data_df)
    print(all_data_df['appid'].value_counts())
    print(all_data_df['steamid'].value_counts())
    print(all_data_df.duplicated().any())

    all_data_df.to_csv(out_data_path, index=False)


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../")     # change working directory
    root = "data/"
    interact_steam_sample_path = root + "steam_interact_sample.csv"
    steam_game_path = root + "steam_game_clean.csv"
    out_steam_data_path = root + "steam_data.csv"
    negative_sample_steam_interact(interact_steam_sample_path, steam_game_path, out_steam_data_path)
    print("Done.")

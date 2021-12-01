import pandas as pd
import os
import sys
import re


def clean_steam(steam_path, out_steam_path):
    steam_df = pd.read_csv(steam_path, parse_dates=['release_date'])

    steam_df.dropna(inplace=True)
    steam_df.drop(columns=['appid', 'release_date', 'developer', 'publisher'], inplace=True)

    cat_df = steam_df['categories'].str.get_dummies(sep=';').add_prefix('cat_')
    # tag_df = steam_df['steamspy_tags'].str.get_dummies(sep=';').add_prefix('tag_')
    steam_df.drop(columns=['categories', 'steamspy_tags'], inplace=True)
    steam_df = pd.concat([steam_df, cat_df], axis=1)

    print("Mapping platforms and genres to dummies")
    steam_df['platforms'].str.replace("||", "|")
    steam_df['genres'].str.replace("||", "|")
    platform_df = steam_df['platforms'].str.get_dummies(sep='|').add_prefix('pf_')
    genre_df = steam_df['genres'].str.get_dummies(sep='|').add_prefix('gn_')
    steam_df.drop(columns=['platforms', 'genres'], inplace=True)
    steam_df = pd.concat([steam_df, platform_df, genre_df], axis=1)

    # steam_df = pd.get_dummies(steam_df, columns=['owners'], prefix=['owner'], drop_first=True)
    steam_df['owners'] = steam_df['owners'].apply(lambda x:
                                                  '0-20000' if x == '0-20000' else
                                                  # '20000-50000' if x == '20000-50000' else
                                                  '20000-200000000')
    steam_df['owners'] = pd.factorize(steam_df['owners'])[0]
    steam_df.dropna(inplace=True)

    # remove non-alphanumeric characters and switch to lower cases
    steam_df['name'] = steam_df['name'].apply(lambda s: re.sub(r'\W', '', s).lower())
    steam_df.drop_duplicates(subset='name', keep='first', inplace=True)

    print("Saving to file, got {} samples".format(len(steam_df.index)))
    steam_df.to_csv(out_steam_path, index=False)
    print("Done")


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/game")  # change working directory
    clean_steam("steam.csv", "steam_clean.csv")

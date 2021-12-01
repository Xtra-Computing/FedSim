import pandas as pd
import os
import sys
import re


def clean_rawg(rawg_path, out_rawg_path):
    print("Loading rawg dataset")
    rawg_df = pd.read_csv(rawg_path, parse_dates=['released', 'updated'])

    print("Dropping unrelated columns")
    rawg_df.drop(columns=['id', 'slug', 'tba', 'metacritic', 'developers', 'publishers',
                          'esrb_rating', 'rating_top', 'ratings_count'], inplace=True)

    print("Mapping website to bool")
    rawg_df['has_website'] = pd.isnull(rawg_df['website'])
    rawg_df.drop(columns=['website'], inplace=True)

    print("Converting date to relative year")
    rawg_df['released_year_before_2020'] = 2020 - rawg_df['released'].dt.year
    rawg_df['updated_year_before_2020'] = 2020 - rawg_df['updated'].dt.year
    rawg_df.drop(columns=['released', 'updated'], inplace=True)

    print("Mapping platforms and genres to dummies")
    rawg_df['platforms'].str.replace("||", "|")
    rawg_df['genres'].str.replace("||", "|")
    platform_df = rawg_df['platforms'].str.get_dummies(sep='|')
    genre_df = rawg_df['genres'].str.get_dummies(sep='|')
    rawg_df.drop(columns=['platforms', 'genres'], inplace=True)
    rawg_df = pd.concat([rawg_df, platform_df, genre_df], axis=1)

    rawg_df.dropna(inplace=True)
    rawg_df = rawg_df[rawg_df['playtime'] < 100]

    # remove non-alphanumeric characters and switch to lower cases
    rawg_df['name'] = rawg_df['name'].apply(lambda s: re.sub(r'\W', '', s).lower())
    rawg_df.drop_duplicates(subset='name', keep='first', inplace=True)

    print("Saving to file, got {} samples".format(len(rawg_df.index)))
    rawg_df.to_csv(out_rawg_path, index=False)
    print("Done")


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/game")  # change working directory
    clean_rawg("rawg.csv", "rawg_clean.csv")

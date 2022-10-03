import pandas as pd

from utils import move_item_to_start_, move_item_to_end_


def load_rawg(rawg_path):
    print("Loading rawg from {}".format(rawg_path))
    rawg_data = pd.read_csv(rawg_path)

    rawg_data.drop(columns=['name'], inplace=True)

    rawg_data.info(verbose=True)

    labels = rawg_data['rating'].to_numpy()
    rawg_data = rawg_data.drop(columns=['rating']).to_numpy()

    return rawg_data, labels


def load_steam(steam_path):
    print("Loading steam from {}".format(steam_path))
    steam_data = pd.read_csv(steam_path)

    steam_data.drop(columns=['name'], inplace=True)

    steam_data.info(verbose=True)

    labels = steam_data['owners'].to_numpy()
    steam_data = steam_data.drop(columns=['owners']).to_numpy()

    return steam_data, labels


def load_both(rawg_path, steam_path, active_party='rawg'):
    print("Loading rawg from {}".format(rawg_path))
    rawg_data = pd.read_csv(rawg_path)
    print("Loading steam from {}".format(steam_path))
    steam_data = pd.read_csv(steam_path)

    if active_party == 'rawg':
        labels = steam_data['owners'].to_numpy()
        steam_data = steam_data.drop(columns=['owners']).to_numpy()

        steam_data.drop(list(steam_data.filter(regex='pf')), axis=1, inplace=True)
        steam_data.drop(list(steam_data.filter(regex='gn')), axis=1, inplace=True)

        # move lon and lat to end
        rawg_cols = list(rawg_data.columns)
        move_item_to_end_(rawg_cols, ['name'])
        rawg_data = rawg_data[rawg_cols]
        print("Current rawg columns {}".format(rawg_data.columns))

        # move lon and lat to start
        steam_cols = list(steam_data.columns)
        move_item_to_start_(steam_cols, ['name'])
        steam_data = steam_data[steam_cols]
        print("Current steam columns {}".format(steam_data.columns))

        data1 = rawg_data.to_numpy()
        data2 = steam_data.to_numpy()
    elif active_party == 'steam':
        labels = steam_data['owners'].to_numpy()
        steam_data = steam_data.drop(columns=['owners'])

        steam_data.drop(list(steam_data.filter(regex='pf')), axis=1, inplace=True)
        steam_data.drop(list(steam_data.filter(regex='gn')), axis=1, inplace=True)

        # move keys to end
        steam_cols = list(steam_data.columns)
        move_item_to_end_(steam_cols, ['name'])
        steam_data = steam_data[steam_cols]
        print("Current steam columns {}".format(steam_data.columns))

        # move lon and lat to start
        rawg_cols = list(rawg_data.columns)
        move_item_to_start_(rawg_cols, ['name'])
        rawg_data = rawg_data[rawg_cols]
        print("Current rawg columns {}".format(rawg_data.columns))

        data1 = steam_data.to_numpy()
        data2 = rawg_data.to_numpy()
    else:
        assert False

    return [data1, data2], labels




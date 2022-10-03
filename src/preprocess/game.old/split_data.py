from sklearn.model_selection import train_test_split
import os
import sys
import pandas as pd


def split_df(data_path, val_rate=0.1, test_rate=0.2, seed=0, save=False):
    """
    Split steam data to train, test, validation set. Output to the same directory as steam_data_path by default.
    Generate 3 new files.
    :param data_path:
    :param val_rate: rate of validation set w.r.t. global dataset
    :param test_rate: rate of test set w.r.t. global dataset
    :param seed: random seed
    :return:
    """
    os.environ['PYTHONHASHSEED'] = str(seed)

    print("Splitting...")
    data_df = pd.read_csv(data_path)
    train_val_df, test_df = train_test_split(data_df, test_size=test_rate, random_state=seed)
    split_val_rate = val_rate / (1. - test_rate)
    train_df, val_df = train_test_split(train_val_df, test_size=split_val_rate, random_state=seed)

    if save:
        base_path = data_path.rsplit('.', 1)[0]
        train_df.to_csv(base_path + "_train.csv", index=False)
        print("Saved to " + base_path + "_train.csv")
        val_df.to_csv(base_path + "_val.csv", index=False)
        print("Saved to " + base_path + "_val.csv")
        test_df.to_csv(base_path + "_test.csv", index=False)
        print("Saved to " + base_path + "_test.csv")

    return train_df, val_df, test_df


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../")  # change working directory
    root = "data/"
    steam_data_path = root + "steam_data.csv"
    split_df(steam_data_path, val_rate=0.1, test_rate=0.2, save=True)

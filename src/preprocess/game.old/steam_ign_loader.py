import pandas as pd


class SteamIgnLoader:
    def __init__(self, steam_train_data_path, steam_val_data_path, steam_test_data_path, ign_data_path):
        self.steam_train_data = pd.read_csv(steam_train_data_path)
        self.steam_val_data = pd.read_csv(steam_val_data_path)
        self.steam_test_data = pd.read_csv(steam_test_data_path)
        self.ign_data = pd.read_csv(ign_data_path)

    def load_parties(self):
        return self.steam_train_data, self.steam_val_data, self.steam_test_data, self.ign_data
import os
import sys
from datetime import datetime
import argparse
import pickle

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import numpy as np

from model.vertical_fl.OnePartyModel import OnePartyModel
from preprocess.hdb import load_hdb, load_both

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory

num_common_features = 2
root = "data/hdb/"
hdb_dataset = root + "hdb_clean.csv"
school_dataset = root + "school_clean.csv"
[X1, X2], y = load_both(hdb_path=hdb_dataset, airbnb_path=school_dataset, active_party='hdb')
name = "hdb_B"
data_cache_path = "cache/hdb_sim.pkl"
print("Loading data from cache")
with open(data_cache_path, 'rb') as f:
    train_dataset, val_dataset, test_dataset, y_scaler, sim_scaler = pickle.load(f)
print("Done")
train_X, train_y, train_idx = train_dataset.top1_dataset
val_X, val_y, val_idx = val_dataset.top1_dataset
test_X, test_y, test_idx = test_dataset.top1_dataset
train_X = train_X[:, X1.shape[1] - num_common_features:]
val_X = val_X[:, X1.shape[1] - num_common_features:]
test_X = test_X[:, X1.shape[1] - num_common_features:]

print("X got {} dimensions".format(train_X.shape[1]))

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()

model = OnePartyModel(model_name=name + "_" + now_string,
                      task='regression',
                      metrics=['r2_score', 'rmse'],
                      n_classes=2,
                      val_rate=0.1,
                      test_rate=0.2,
                      device='cuda:{}'.format(args.gpu),
                      hidden_sizes=[400, 200],
                      train_batch_size=4096,
                      test_batch_size=4096,
                      num_epochs=200,
                      learning_rate=1e-2,
                      weight_decay=1e-5,
                      num_workers=4 if sys.gettrace() is None else 0,
                      use_scheduler=False,
                      sche_factor=0.1,
                      sche_patience=10,
                      sche_threshold=0.0001,
                      writer_path="runs/{}_{}".format(name, now_string),
                      model_save_path="ckp/{}_{}.pth".format(name, now_string)
                      )
model._train(train_X, val_X, test_X, train_y, val_y, test_y, y_scaler=y_scaler)

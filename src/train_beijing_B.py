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
from preprocess.beijing import load_house, load_both

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory

root = "data/beijing/"
house_dataset = root + "house_clean.csv"
airbnb_dataset = root + "airbnb_clean.csv"

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--leak-p', type=float, default=1.0)
parser.add_argument('-g', '--gpu', type=int, default=3)
args = parser.parse_args()

num_common_features = 2
[X1, X2], y = load_both(house_path=house_dataset, airbnb_path=airbnb_dataset, active_party='house')
data_cache_path = "cache/beijing_sim.pkl"
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

print("X got {} dimensions".format(X2.shape[1]))
name = "beijing_B"
# reg = LinearRegression().fit(X, y)
# score = np.sqrt(metrics.mean_squared_error(reg.predict(X), y))
# print(score)

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
                      num_epochs=100,
                      learning_rate=3e-3,
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


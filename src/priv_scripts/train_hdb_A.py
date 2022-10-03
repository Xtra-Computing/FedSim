import os
import sys
from datetime import datetime
import argparse

os.chdir(sys.path[0] + "/../../")  # change working directory
sys.path.append(os.path.join(os.getcwd(), "src"))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import numpy as np

from model.vertical_fl.OnePartyModel import OnePartyModel
from preprocess.hdb import load_hdb

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

root = "data/hdb/"
dataset = "hdb_clean.csv"

X, y = load_hdb(root + dataset)
print("X got {} dimensions".format(X.shape[1]))
name = "hdb_A"

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
model.train_single(X, y, scale=True)

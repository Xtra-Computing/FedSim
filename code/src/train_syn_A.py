import os
import sys
import argparse
from datetime import datetime

from model.vertical_fl.OnePartyModel import OnePartyModel
from preprocess.sklearn.syn_data_generator import TwoPartyClsMany2ManyGenerator

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--noise-scale', type=float, default=0.0)
parser.add_argument('-p', '--leak-p', type=float, default=1.0)
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/"
num_common_features = 5
noise_scale = args.noise_scale

syn_generator = TwoPartyClsMany2ManyGenerator.from_pickle(
    root + "syn_cls_many2many_generator_noise_{:.1f}.pkl".format(noise_scale))
(X1, X2), y = syn_generator.get_parties()
X = X1[:, :-num_common_features]
# X = X1
print("X got {} dimensions".format(X.shape[1]))
name = "syn_a_noise_scale_{:.1f}".format(noise_scale)

model = OnePartyModel(model_name=name + "_" + now_string,
                      task='binary_cls',
                      metrics=['accuracy'],
                      n_classes=2,
                      val_rate=0.1,
                      test_rate=0.2,
                      device='cuda:{}'.format(args.gpu),
                      hidden_sizes=[100, 100],
                      train_batch_size=4096,
                      test_batch_size=4096,
                      num_epochs=100,
                      learning_rate=1e-3,
                      weight_decay=1e-5,
                      num_workers=4 if sys.gettrace() is None else 0,
                      use_scheduler=False,
                      sche_factor=0.1,
                      sche_patience=10,
                      sche_threshold=0.0001,
                      writer_path="runs/{}_{}".format(name, now_string),
                      model_save_path="ckp/{}_{}.pth".format(name, now_string)
                      )
model.train_all(X, y)

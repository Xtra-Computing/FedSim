import os
import sys
from datetime import datetime
import argparse

from model.vertical_fl.FedSimModel import FedSimModel
from preprocess.nytaxi.ny_loader import NYBikeTaxiLoader

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/nytaxi/"
bike_dataset = "bike_201606_clean_sample_2e5.pkl"
taxi_dataset = "taxi_201606_clean_sample_1e5.pkl"
# taxi_dataset = "taxi_201606_clean.csv"

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--leak-p', type=float, default=1.0)
parser.add_argument('-g', '--gpu', type=int, default=0)
parser.add_argument('-k', '--top-k', type=int, default=None)
args = parser.parse_args()

num_common_features = 4
data_loader = NYBikeTaxiLoader(bike_path=root + bike_dataset, taxi_path=root + taxi_dataset, link=True)
[X1, X2], y = data_loader.load_parties()
name = "ny_fedsim"

model = FedSimModel(num_common_features=num_common_features,
                    raw_output_dim=10,
                    feature_wise_sim=False,
                    task='regression',
                    metrics=['r2_score', 'rmse'],
                    dataset_type='real',
                    blocking_method='knn',
                    n_classes=2,
                    grid_min=-10.0,
                    grid_max=10.0,
                    grid_width=1.5,
                    knn_k=50,
                    filter_top_k=args.top_k,
                    kd_tree_radius=2e-3,
                    tree_leaf_size=1000,
                    model_name=name + "_" + now_string,
                    val_rate=0.1,
                    test_rate=0.2,
                    drop_key=True,
                    device='cuda:{}'.format(args.gpu),
                    hidden_sizes=[200, 100],
                    train_batch_size=64,
                    test_batch_size=4096,
                    num_epochs=20,
                    learning_rate=3e-4,
                    weight_decay=1e-5,
                    sim_learning_rate=3e-4,
                    sim_weight_decay=1e-5,
                    sim_batch_size=4096,
                    update_sim_freq=1,
                    num_workers=8 if sys.gettrace() is None else 0,
                    use_scheduler=False, sche_factor=0.1, sche_patience=10, sche_threshold=0.0001,
                    writer_path="runs/{}_{}".format(name, now_string),
                    model_save_path="ckp/{}_{}.pth".format(name, now_string),
                    sim_model_save_path="ckp/{}_{}_sim.pth".format(name, now_string),
                    # SplitNN parameters
                    local_hidden_sizes=[[100], [100]],
                    agg_hidden_sizes=[100],
                    cut_dims=[50, 50],

                    # fedsim parameters
                    use_conv=True,
                    merge_hidden_sizes=[400],
                    sim_hidden_sizes=[10],
                    merge_model_save_path="ckp/{}_{}_merge.pth".format(name, now_string),
                    merge_dropout_p=0.3,
                    conv_n_channels=8,
                    conv_kernel_v_size=5,

                    # private link parameters
                    link_epsilon=1e-1,
                    link_delta=1e-1,
                    link_threshold_t=2e-2,
                    sim_leak_p=args.leak_p,
                    link_n_jobs=-1,
                    )
model.train_splitnn(X1, X2, y, data_cache_path="cache/ny_sim.pkl", scale=True)

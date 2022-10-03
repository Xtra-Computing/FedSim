import os
import sys
import abc
import pickle
import random
from datetime import datetime
import gc

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import torch_optimizer as adv_optim
from torchsummaryX import summary
from tqdm import tqdm
import deprecation

from .SimModel import SimModel
from model.base import MLP, SplitNN
import metric
import metric.base


# class Top1GroupDataset(Dataset):
#     def __init__(self, data, labels, data_idx):
#         self.raw_data_labels = np.concatenate([data, labels], axis=1)
#
#         self.filtered_data = {}
#         for i in data_idx.shape[0]:
#             prev_sim_score = self.filtered_data[data_idx[i]][0]
#             cur_sim_score = self.raw_data_labels[data_idx[i]][0]
#             if not (data_idx[i] in self.filtered_data and cur_sim_score < prev_sim_score):
#                 self.filtered_data[data_idx[i]] = self.raw_data_labels[data_idx[i]]
#
#         self.data_idx = np.array(list(self.filtered_data.keys()))
#         self.data_labels = np.array(list(self.filtered_data.values()))
#         self.data = self.data_labels[:, :-1]
#         self.labels = self.data_labels[:, -1]
#
#     def __len__(self):
#         return self.data.shape[0]
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         return self.data[idx], self.labels[idx], self.data_idx[idx]


class Top1SimModel(SimModel):
    def __init__(self, num_common_features, **kwargs):
        super().__init__(num_common_features, **kwargs)

        self.data1_shape = None
        self.data2_shape = None

    def prepare_train_combine(self, data1, data2, labels, data_cache_path=None, scale=False):
        train_dataset, val_dataset, test_dataset, y_scaler = \
            super().prepare_train_combine(data1, data2, labels, data_cache_path, scale)
        train_X, train_y, train_idx = train_dataset.top1_dataset
        val_X, val_y, val_idx = val_dataset.top1_dataset
        test_X, test_y, test_idx = test_dataset.top1_dataset
        return train_X, val_X, test_X, train_y, val_y, test_y, train_idx, val_idx, test_idx, y_scaler

    @deprecation.deprecated()
    def __prepare_train_combine(self, data1, data2, labels, data_cache_path=None, scale=False):
        if data_cache_path and os.path.isfile(data_cache_path):
            print("Loading data from cache")
            with open(data_cache_path, 'rb') as f:
                train_X, val_X, test_X, train_y, val_y, test_y, train_idx, val_idx, test_idx, y_scaler = pickle.load(f)
        else:
            print("Splitting data")
            train_data1, val_data1, test_data1, train_labels, val_labels, test_labels, train_idx1, val_idx1, test_idx1 = \
                self.split_data(data1, labels, val_rate=self.val_rate, test_rate=self.test_rate)
            if self.dataset_type == 'syn':
                train_data2 = data2[train_idx1]
                val_data2 = data2[val_idx1]
                test_data2 = data2[test_idx1]
            elif self.dataset_type == 'real':
                train_data2 = data2
                val_data2 = data2
                test_data2 = data2
            else:
                assert False, "Not supported dataset type"
            print("Matching training set")
            self.sim_scaler = None  # scaler will fit train_Xs and transform val_Xs, test_Xs
            preserve_key = not self.drop_key
            train_Xs, train_y, train_idx = self.match(train_data1, train_data2, train_labels, idx=train_idx1,
                                                      preserve_key=preserve_key, grid_min=self.grid_min,
                                                      grid_max=self.grid_max, grid_width=self.grid_width,
                                                      knn_k=self.knn_k, tree_leaf_size=self.tree_leaf_size,
                                                      radius=self.tree_radius)
            print("Matching validation set")
            val_Xs, val_y, val_idx = self.match(val_data1, val_data2, val_labels, idx=val_idx1,
                                                preserve_key=preserve_key,
                                                grid_min=self.grid_min, grid_max=self.grid_max,
                                                grid_width=self.grid_width, knn_k=self.knn_k,
                                                tree_leaf_size=self.tree_leaf_size,
                                                radius=self.tree_radius)
            print("Matching test set")
            test_Xs, test_y, test_idx = self.match(test_data1, test_data2, test_labels, idx=test_idx1,
                                                   preserve_key=preserve_key, grid_min=self.grid_min,
                                                   grid_max=self.grid_max, grid_width=self.grid_width,
                                                   knn_k=self.knn_k,
                                                   tree_leaf_size=self.tree_leaf_size,
                                                   radius=self.tree_radius)

            # remove sim_scores in data2
            train_Xs[1] = train_Xs[1][:, 1:]
            val_Xs[1] = val_Xs[1][:, 1:]
            test_Xs[1] = test_Xs[1][:, 1:]

            train_X = np.concatenate(train_Xs, axis=1)
            val_X = np.concatenate(val_Xs, axis=1)
            test_X = np.concatenate(test_Xs, axis=1)

            print("Replace NaN with mean value")
            col_mean = np.nanmean(train_X, axis=0)
            train_indices = np.where(np.isnan(train_X))
            train_X[train_indices] = np.take(col_mean, train_indices[1])
            print("Train done.")
            val_indices = np.where(np.isnan(val_X))
            val_X[val_indices] = np.take(col_mean, val_indices[1])
            print("Validation done.")
            test_indices = np.where(np.isnan(test_X))
            test_X[test_indices] = np.take(col_mean, test_indices[1])
            print("Test done.")

            train_X, train_y, train_idx = self._group(train_X, train_y.reshape(-1, 1), train_idx)
            val_X, val_y, val_idx = self._group(val_X, val_y.reshape(-1, 1), val_idx)
            test_X, test_y, test_idx = self._group(test_X, test_y.reshape(-1, 1), test_idx)

            y_scaler = None
            if scale:
                x_scaler = StandardScaler()
                train_X = x_scaler.fit_transform(train_X)
                val_X = x_scaler.transform(val_X)
                test_X = x_scaler.transform(test_X)

            if scale and self.task == 'regression':
                y_scaler = MinMaxScaler(feature_range=(0, 1))
                train_y = y_scaler.fit_transform(train_y.reshape(-1, 1)).flatten()
                val_y = y_scaler.transform(val_y.reshape(-1, 1)).flatten()
                test_y = y_scaler.transform(test_y.reshape(-1, 1)).flatten()

            if data_cache_path:
                print("Saving data to cache")
                with open(data_cache_path, 'wb') as f:
                    pickle.dump([train_X, val_X, test_X, train_y, val_y, test_y, train_idx, val_idx, test_idx, y_scaler], f)

        return train_X, val_X, test_X, train_y, val_y, test_y, train_idx, val_idx, test_idx, y_scaler

    @staticmethod
    @deprecation.deprecated()
    def _group(data, labels, data_idx):
        assert data.shape[0] == labels.shape[0] == data_idx.shape[0]
        print("Start grouping, got {} samples".format(data_idx.shape[0]))
        raw_data_labels = np.concatenate([data, labels], axis=1)
        filtered_data = {}
        filtered_data_idx = {}
        for i in range(data_idx.shape[0]):
            cur_sim_score = raw_data_labels[i][0]
            idx1 = data_idx[i][0]
            if not (idx1 in filtered_data and cur_sim_score < filtered_data[idx1][0]):
                filtered_data[idx1] = raw_data_labels[i]
                filtered_data_idx[idx1] = data_idx[i][1]

        print("Finished grouping, got {} samples".format(len(filtered_data)))
        print("Exact matched rate {}".format(np.average([abs(k-v)<1e-7 for k, v in filtered_data_idx.items()])))
        print("Non-NaN rate {}".format(np.average(np.invert(np.isnan(list(filtered_data_idx.values()))))))
        grouped_data_idx = np.array(list(filtered_data.keys()))
        data_labels = np.array(list(filtered_data.values()))
        grouped_data = data_labels[:, 1:-1]
        grouped_labels = data_labels[:, -1]
        return grouped_data, grouped_labels, grouped_data_idx

    def train_combine(self, data1, data2, labels, data_cache_path=None, scale=False):
        train_X, val_X, test_X, train_y, val_y, test_y, train_idx, val_idx, test_idx, y_scaler = \
            self.prepare_train_combine(data1, data2, labels, data_cache_path, scale)

        return self._train(train_X, val_X, test_X, train_y, val_y, test_y,
                           y_scaler=y_scaler)

    def train_splitnn(self, data1, data2, labels, data_cache_path=None, scale=False):
        start_time = datetime.now()
        train_X, val_X, test_X, train_y, val_y, test_y, train_idx, val_idx, test_idx, y_scaler = \
            self.prepare_train_combine(data1, data2, labels, data_cache_path, scale)
        time_duration_sec = (datetime.now() - start_time).seconds
        print("Preparing time (sec): {}".format(time_duration_sec))

        start_time = datetime.now()
        print("Loading data")
        if train_idx is None:
            train_dataset = TensorDataset(train_X.float(), train_y.float())
        else:  # need to calculate final accuracy
            train_dataset = TensorDataset(train_X.float(), train_y.float(), torch.tensor(train_idx).int())
        # IMPORTANT: Set num_workers to 0 to prevent deadlock on RTX3090 for unknown reason.
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                  num_workers=0)

        print("Prepare for training")
        self.data1_shape = data1.shape
        self.data2_shape = data2.shape
        input_dims = [self.data1_shape[1] - self.num_common_features,
                      self.data2_shape[1] - self.num_common_features]
        num_parties = 2
        if self.drop_key:
            num_features = data1.shape[1] + data2.shape[1] - 2 * self.num_common_features
        else:
            num_features = data1.shape[1] + data2.shape[1]

        del data1, data2
        gc.collect()

        if self.task == 'binary_cls':
            output_dim = 1
            local_models = [MLP(input_size=input_dims[i], hidden_sizes=self.local_hidden_sizes[i],
                                output_size=self.cut_dims[i], activation=None) for i in range(num_parties)]
            agg_model = MLP(input_size=sum(self.cut_dims), hidden_sizes=self.agg_hidden_sizes,
                            output_size=output_dim, activation='sigmoid')
            self.model = SplitNN(local_models, input_dims, agg_model)
            criterion = nn.BCELoss()
            val_criterion = nn.BCELoss()
        elif self.task == 'multi_cls':
            output_dim = self.n_classes
            local_models = [MLP(input_size=input_dims[i], hidden_sizes=self.local_hidden_sizes[i],
                                output_size=self.cut_dims[i], activation=None) for i in range(num_parties)]
            agg_model = MLP(input_size=sum(self.cut_dims), hidden_sizes=self.agg_hidden_sizes,
                            output_size=output_dim, activation=None)
            self.model = SplitNN(local_models, input_dims, agg_model)
            criterion = nn.CrossEntropyLoss()
            val_criterion = nn.CrossEntropyLoss()
        elif self.task == 'regression':
            output_dim = 1
            local_models = [MLP(input_size=input_dims[i], hidden_sizes=self.local_hidden_sizes[i],
                                output_size=self.cut_dims[i], activation=None) for i in range(num_parties)]
            agg_model = MLP(input_size=sum(self.cut_dims), hidden_sizes=self.agg_hidden_sizes,
                            output_size=output_dim, activation='sigmoid')
            self.model = SplitNN(local_models, input_dims, agg_model)
            criterion = nn.MSELoss()
            val_criterion = nn.MSELoss()
        else:
            assert False, "Unsupported task"
        self.model = self.model.to(self.device)

        optimizer = adv_optim.Lamb(self.model.parameters(),
                                   lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.sche_factor,
                                                             patience=self.sche_patience,
                                                             threshold=self.sche_threshold)
        else:
            scheduler = None

        best_train_metric_scores = [m.worst for m in self.metrics_f]
        best_val_metric_scores = [m.worst for m in self.metrics_f]
        best_test_metric_scores = [m.worst for m in self.metrics_f]
        if train_idx is not None:
            answer_all = dict(zip(train_idx, train_y))
        print("Start training")
        summary(self.model, torch.zeros([self.train_batch_size, num_features]).to(self.device))
        print(str(self))
        for epoch in range(self.num_epochs):
            if train_idx is not None:
                train_pred_all = {}
            # train
            train_loss = 0.0
            n_train_batches = 0
            self.model.train()
            all_preds = np.zeros((0, 1))
            all_labels = np.zeros(0)
            for info in tqdm(train_loader, desc="Train"):
                if train_idx is not None:
                    data, labels, idx = info
                else:
                    data, labels = info
                data = data.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(data)
                if self.task == 'binary_cls':
                    outputs = outputs.flatten()
                    loss = criterion(outputs, labels)
                    preds = outputs > 0.5
                    preds = preds.reshape(-1, 1).detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                elif self.task == 'multi_cls':
                    loss = criterion(outputs, labels.long())
                    preds = torch.argmax(outputs, dim=1)
                    preds = preds.reshape(-1, 1).detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                elif self.task == 'regression':
                    outputs = outputs.flatten()
                    loss = criterion(outputs, labels)
                    preds = outputs
                    preds = preds.reshape(-1, 1).detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                    if y_scaler is not None:
                        preds = y_scaler.inverse_transform(preds.reshape(-1, 1))
                        labels = y_scaler.inverse_transform(labels.reshape(-1, 1)).flatten()
                else:
                    assert False, "Unsupported task"

                loss.backward()
                optimizer.step()

                all_preds = np.concatenate([all_preds, preds])
                all_labels = np.concatenate([all_labels, labels])
                n_train_batches += 1

            train_loss /= n_train_batches

            train_metric_scores = []
            for metric_f in self.metrics_f:
                train_metric_scores.append(metric_f(all_preds, all_labels))

            # validation and test
            val_loss, val_metric_scores = self.eval_score(val_X, val_y, loss_criterion=criterion,
                                                          name='Val', y_scaler=y_scaler)
            test_loss, test_metric_scores = self.eval_score(test_X, test_y, loss_criterion=criterion,
                                                            name='Test', y_scaler=y_scaler)
            if self.use_scheduler:
                scheduler.step(val_loss)

            # The first metric determines early stopping
            if self.metrics[0] in ['accuracy', 'r2_score']:
                is_best = (val_metric_scores[0] > best_val_metric_scores[0])
            elif self.metrics[0] in ['rmse']:
                is_best = (val_metric_scores[0] < best_val_metric_scores[0])
            else:
                assert False, "Unsupported metric"

            if is_best:
                best_train_metric_scores = train_metric_scores
                best_val_metric_scores = val_metric_scores
                best_test_metric_scores = test_metric_scores
                if self.model_save_path is not None:
                    torch.save(self.model.state_dict(), self.model_save_path)

            print("Epoch {}: {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}"
                  .format(epoch + 1, "Loss:", train_loss, val_loss, test_loss))
            self.writer.add_scalars('Loss', {'Train': train_loss,
                                             'Validation': val_loss,
                                             'Test': test_loss}, epoch + 1)
            for i in range(len(self.metrics)):
                print("          {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}"
                      .format(self.metrics_f[i].name, train_metric_scores[i],
                              val_metric_scores[i], test_metric_scores[i]))
                self.writer.add_scalars(self.metrics_f[i].name, {'Train': train_metric_scores[i],
                                                                 'Validation': val_metric_scores[i],
                                                                 'Test': test_metric_scores[i]}, epoch + 1)
            print("Best:")
            for i in range(len(self.metrics)):
                print("          {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}"
                      .format(self.metrics_f[i].name, best_train_metric_scores[i],
                              best_val_metric_scores[i], best_test_metric_scores[i]))

        time_duration_sec = (datetime.now() - start_time).seconds
        print("Training time (sec): {}".format(time_duration_sec))

        return best_test_metric_scores

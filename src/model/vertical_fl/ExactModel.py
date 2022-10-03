import os
import abc
import pickle
import warnings
import gc
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KDTree, BallTree
import joblib

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import torch_optimizer as adv_optim
from torchsummaryX import summary
from tqdm import tqdm

from tqdm import tqdm
import deprecation
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from captum.attr import IntegratedGradients
from lhbf import BloomFilter
from nltk import ngrams
import faiss

from .SimModel import SimModel
from model.base import MLP, SplitNN
from .TwoPartyModel import TwoPartyBaseModel
from utils import scaled_edit_distance


class ExactModel(TwoPartyBaseModel):
    def __init__(self, num_common_features,
                 # Split Learning
                 local_hidden_sizes=None, agg_hidden_sizes=None, cut_dims=None,

                 **kwargs):
        super().__init__(num_common_features, **kwargs)
        self.local_hidden_sizes = local_hidden_sizes
        self.agg_hidden_sizes = agg_hidden_sizes
        self.cut_dims = cut_dims

        self.data1_shape = None
        self.data2_shape = None

    def match(self, data1, data2, labels, idx=None, preserve_key=False, **kwargs):
        # extract common features from data
        col1 = ["A{}".format(i) if i < data1.shape[1] - self.num_common_features
                else "key{}".format(i - data1.shape[1] + self.num_common_features)
                for i in range(data1.shape[1])]
        col2 = ["B{}".format(i - self.num_common_features) if i >= self.num_common_features
                else "key{}".format(i)
                for i in range(data2.shape[1])]

        data1_df = pd.DataFrame(data=data1, columns=col1)
        data1_df['label'] = pd.DataFrame(labels)
        data2_df = pd.DataFrame(data=data2, columns=col2)
        # data1_df.set_index(["key{}".format(i) for i in range(self.num_common_features)], inplace=True)
        # data2_df.set_index(["key{}".format(i) for i in range(self.num_common_features)], inplace=True)

        linked_cols = ["key{}".format(i) for i in range(self.num_common_features)]
        exact_linked_data_df = data1_df.merge(data2_df, how='left', left_on=linked_cols, right_on=linked_cols)
        print("Got {} pairs, {} of which is null, linkage rate {:.2f}%"
              .format(len(exact_linked_data_df.index), exact_linked_data_df.B1.isnull().sum(),
                      100 - exact_linked_data_df.B0.isnull().sum() / len(exact_linked_data_df.index) * 100))

        exact_linked_data_df.drop(columns=linked_cols, inplace=True)
        matched_labels = exact_linked_data_df['label'].to_numpy()
        matched_data = exact_linked_data_df.drop(columns=['label']).to_numpy().astype('float')

        return matched_data, matched_labels

    def prepare_train(self, data1, data2, labels, data_cache_path=None, scale=False):
        if data_cache_path and os.path.isfile(data_cache_path):
            print("Loading data from cache")
            with open(data_cache_path, 'rb') as f:
                train_dataset, val_dataset, test_dataset, y_scaler = pickle.load(f)
            print("Done")
        else:
            print("Splitting data")
            train_data1, val_data1, test_data1, train_labels, val_labels, test_labels, train_idx1, val_idx1, test_idx1 = \
                self.split_data(data1, labels, val_rate=self.val_rate, test_rate=self.test_rate)

            preserve_key = not self.drop_key
            print("Matching training dataset")
            train_X, train_y = self.match(train_data1, data2, train_labels,
                                          preserve_key=preserve_key)
            print("Matching validation set")
            val_X, val_y = self.match(val_data1, data2, val_labels,
                                      preserve_key=preserve_key)
            print("Matching test set")
            test_X, test_y = self.match(test_data1, data2, test_labels,
                                        preserve_key=preserve_key)

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

            if scale:
                print("Scaling X")
                x_scaler = StandardScaler()
                train_X = x_scaler.fit_transform(train_X)
                val_X = x_scaler.transform(val_X)
                test_X = x_scaler.transform(test_X)
                print("Scale done.")

            y_scaler = None
            if scale and self.task == 'regression':
                print("Scaling y")
                y_scaler = MinMaxScaler(feature_range=(0, 1))
                train_y = y_scaler.fit_transform(train_y.reshape(-1, 1)).flatten()
                val_y = y_scaler.transform(val_y.reshape(-1, 1)).flatten()
                test_y = y_scaler.transform(test_y.reshape(-1, 1)).flatten()
                print("Scale done")

            train_dataset = TensorDataset(torch.tensor(train_X).float(), torch.tensor(train_y).float())
            val_dataset = TensorDataset(torch.tensor(val_X).float(), torch.tensor(val_y).float())
            test_dataset = TensorDataset(torch.tensor(test_X).float(), torch.tensor(test_y).float())

            if data_cache_path:
                print("Saving data to cache")
                with open(data_cache_path, 'wb') as f:
                    pickle.dump([train_dataset, val_dataset, test_dataset, y_scaler], f)
                print("Saved")

        return train_dataset, val_dataset, test_dataset, y_scaler

    def train_splitnn(self, data1, data2, labels, data_cache_path=None, scale=False):
        start_time = datetime.now()
        train_dataset, val_dataset, test_dataset, y_scaler = \
            self.prepare_train(data1, data2, labels, data_cache_path, scale)
        time_duration_sec = (datetime.now() - start_time).seconds
        print("Preparing time (sec): {}".format(time_duration_sec))

        start_time = datetime.now()
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                  num_workers=self.num_workers, multiprocessing_context=self.multiprocess_context)
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

        print("Start training")
        summary(self.model, torch.zeros([self.train_batch_size, num_features]).to(self.device))
        print(str(self))
        for epoch in range(self.num_epochs):
            # train
            train_loss = 0.0
            n_train_batches = 0
            self.model.train()
            all_preds = np.zeros((0, 1))
            all_labels = np.zeros(0)
            for info in tqdm(train_loader, desc="Train"):
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

                train_loss += loss.item()

                all_preds = np.concatenate([all_preds, preds])
                all_labels = np.concatenate([all_labels, labels])
                n_train_batches += 1

            train_loss /= n_train_batches

            train_metric_scores = []
            for metric_f in self.metrics_f:
                train_metric_scores.append(metric_f(all_preds, all_labels))

            # validation and test
            val_loss, val_metric_scores = self.eval_merge_score(val_dataset, loss_criterion=criterion,
                                                                name='Val', y_scaler=y_scaler)
            test_loss, test_metric_scores = self.eval_merge_score(test_dataset, loss_criterion=criterion,
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

    def eval_merge_score(self, val_dataset, loss_criterion=None, name='Val', y_scaler=None):
        assert self.model is not None, "Model has not been initialized"

        # IMPORTANT: Set num_workers of val_loader to 0 to prevent deadlock on RTX3090 for unknown reason.
        #            multiprocessing_context should also be set to default.
        # val_loader = DataLoader(val_dataset, batch_size=self.test_batch_size, shuffle=False,
        #                         num_workers=self.num_workers, multiprocessing_context=self.multiprocess_context)
        val_loader = DataLoader(val_dataset, batch_size=self.test_batch_size, shuffle=False,
                                num_workers=0)
        val_loss = 0.0
        n_val_batches = 0
        # output_dim = 1 if self.task in ['binary_cls', 'regression'] else self.n_classes
        all_preds = np.zeros((0, 1))
        all_labels = np.zeros(0)

        with torch.no_grad():
            self.model.eval()
            for info in tqdm(val_loader, desc=name):
                data, labels = info

                data = data.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(data)

                if self.task == 'binary_cls':
                    outputs = outputs.flatten()
                    if loss_criterion is not None:
                        loss = loss_criterion(outputs, labels)
                        val_loss += loss.item()
                    preds = outputs > 0.5
                    preds = preds.reshape(-1, 1).detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                elif self.task == 'multi_cls':
                    if loss_criterion is not None:
                        loss = loss_criterion(outputs, labels.long())
                        val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    preds = preds.reshape(-1, 1).detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                elif self.task == 'regression':
                    outputs = outputs.flatten()
                    if loss_criterion is not None:
                        loss = loss_criterion(outputs, labels)
                        val_loss += loss.item()
                    preds = outputs
                    preds = preds.reshape(-1, 1).detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                    if y_scaler is not None:
                        preds = y_scaler.inverse_transform(preds.reshape(-1, 1))
                        labels = y_scaler.inverse_transform(labels.reshape(-1, 1)).flatten()
                else:
                    assert False, "Unsupported task"

                all_preds = np.concatenate([all_preds, preds])
                all_labels = np.concatenate([all_labels, labels])
                n_val_batches += 1

        if loss_criterion is not None:
            val_loss /= n_val_batches
        else:
            val_loss = -1.

        metric_scores = []
        for metric_f in self.metrics_f:
            metric_scores.append(metric_f(all_preds, all_labels))

        return val_loss, metric_scores

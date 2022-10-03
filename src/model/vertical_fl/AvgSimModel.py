import os
import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from torchsummaryX import summary
import torch_optimizer as adv_optim

from .SimModel import SimModel
from model.base.MLP import MLP


class AvgDataset(Dataset):
    def __init__(self, data1, data2, labels, data_idx, sim_dim=1):
        assert data1.shape[0] == data2.shape[0] == data_idx.shape[0]
        # remove similarity scores in data1 (at column 0)
        data1_labels = np.concatenate([data1[:, sim_dim:], labels.reshape(-1, 1)], axis=1)

        print("Grouping data")
        grouped_data1 = {}
        grouped_data2 = {}
        for i in range(data_idx.shape[0]):
            idx1, idx2 = data_idx[i]
            new_data2 = np.concatenate([idx2.reshape(1, 1), data2[i].reshape(1, -1)], axis=1)
            if idx1 in grouped_data2:
                grouped_data2[idx1] = np.concatenate([grouped_data2[idx1], new_data2], axis=0)
            else:
                grouped_data2[idx1] = new_data2
            grouped_data1[idx1] = data1_labels[i]
        print("Done")

        group1_data_idx = np.array(list(grouped_data1.keys()))
        group1_data1_labels = np.array(list(grouped_data1.values()))
        group2_data_idx = np.array(list(grouped_data2.keys()))
        group2_data2 = np.array(list(grouped_data2.values()), dtype='object')

        print("Sorting data")
        group1_order = group1_data_idx.argsort()
        group2_order = group2_data_idx.argsort()

        group1_data_idx = group1_data_idx[group1_order]
        group1_data1_labels = group1_data1_labels[group1_order]
        group2_data_idx = group2_data_idx[group2_order]
        group2_data2 = group2_data2[group2_order]
        assert (group1_data_idx == group2_data_idx).all()
        print("Done")

        self.data1_idx: torch.Tensor = torch.from_numpy(group1_data_idx)
        self.data1: torch.Tensor = torch.from_numpy(group1_data1_labels[:, :-1])
        self.data1_labels: torch.Tensor = torch.from_numpy(group1_data1_labels[:, -1])
        data2: list = group2_data2

        final_data = []
        final_weights = []
        final_labels = []
        final_idx = []
        print("Retrieve data")
        for i in range(self.data1.shape[0]):
            d1 = self.data1[i]
            for j in range(data2[i].shape[0]):
                d2 = torch.from_numpy(data2[i][j].astype(np.float))
                idx2 = d2[0].item()
                d2 = d2[1:]  # remove index
                weight = 1 / data2[i].shape[0]
                # d_line: sim_score, data1, data2
                d_line = torch.cat([d2[:sim_dim], d1, d2[sim_dim:]], dim=0)

                final_data.append(d_line)
                final_weights.append(weight)
                final_labels.append(self.data1_labels[i])
                final_idx.append((self.data1_idx[i], idx2))
        print("Done")

        self.data = torch.stack(final_data)
        self.weights = torch.tensor(final_weights)
        self.labels = torch.tensor(final_labels)
        self.data_idx = torch.tensor(final_idx)

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx], self.labels[idx], self.weights[idx], self.data_idx[idx, 0], self.data_idx[idx, 1]


class AvgSimModel(SimModel):
    def __init__(self, num_common_features, merge_mode='avg', **kwargs):
        super().__init__(num_common_features, **kwargs)
        assert merge_mode in ['avg']
        self.merge_mode = merge_mode

        self.data1_shape = None
        self.data2_shape = None

    @staticmethod
    def var_collate_fn(batch):
        data1 = torch.stack([item[0] for item in batch])
        data2 = [item[1] for item in batch]
        labels = torch.stack([item[2] for item in batch])
        idx = torch.stack([item[3] for item in batch])
        return data1, data2, labels, idx

    def prepare_train_combine(self, data1, data2, labels, data_cache_path=None, scale=False):
        if data_cache_path and os.path.isfile(data_cache_path):
            print("Loading data from cache")
            with open(data_cache_path, 'rb') as f:
                train_dataset, val_dataset, test_dataset, y_scaler = pickle.load(f)
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
            assert self.sim_scaler is not None
            print("Matching validation set")
            val_Xs, val_y, val_idx = self.match(val_data1, val_data2, val_labels, idx=val_idx1,
                                                preserve_key=preserve_key, grid_min=self.grid_min,
                                                grid_max=self.grid_max, grid_width=self.grid_width, knn_k=self.knn_k,
                                                tree_leaf_size=self.tree_leaf_size, radius=self.tree_radius)
            assert self.sim_scaler is not None
            print("Matching test set")
            test_Xs, test_y, test_idx = self.match(test_data1, test_data2, test_labels, idx=test_idx1,
                                                   preserve_key=preserve_key, grid_min=self.grid_min,
                                                   grid_max=self.grid_max, grid_width=self.grid_width, knn_k=self.knn_k,
                                                   tree_leaf_size=self.tree_leaf_size, radius=self.tree_radius)

            for train_X, val_X, test_X in zip(train_Xs, val_Xs, test_Xs):
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
                    x_scaler = StandardScaler()
                    train_X[:] = x_scaler.fit_transform(train_X)
                    val_X[:] = x_scaler.transform(val_X)
                    test_X[:] = x_scaler.transform(test_X)

            y_scaler = None
            if scale:
                y_scaler = MinMaxScaler(feature_range=(0, 1))
                train_y = y_scaler.fit_transform(train_y.reshape(-1, 1)).flatten()
                val_y = y_scaler.transform(val_y.reshape(-1, 1)).flatten()
                test_y = y_scaler.transform(test_y.reshape(-1, 1)).flatten()

            train_dataset = AvgDataset(train_Xs[0], train_Xs[1], train_y, train_idx)
            val_dataset = AvgDataset(val_Xs[0], val_Xs[1], val_y, val_idx)
            test_dataset = AvgDataset(test_Xs[0], test_Xs[1], test_y, test_idx)

            if data_cache_path:
                print("Saving data to cache")
                with open(data_cache_path, 'wb') as f:
                    pickle.dump([train_dataset, val_dataset, test_dataset, y_scaler], f)

        return train_dataset, val_dataset, test_dataset, y_scaler

    def merge_pred(self, pred_all: list, idx=None):
        pred_array = np.array(pred_all).T
        avg_pred = np.average(pred_array[0])
        if self.task == 'binary_cls':
            return avg_pred > 0.5
        elif self.task == 'regression':
            return avg_pred
        else:
            assert False, "Not Implemented"

    def train_combine(self, data1, data2, labels, data_cache_path=None, scale=False):
        train_dataset, val_dataset, test_dataset, y_scaler = \
            self.prepare_train_combine(data1, data2, labels, data_cache_path, scale=scale)

        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                  num_workers=self.num_workers, multiprocessing_context=self.multiprocess_context)
        num_features = next(iter(train_loader))[0].shape[1] - 1
        self.data1_shape = data1.shape
        self.data2_shape = data2.shape

        print("Prepare for training")
        if self.task == 'binary_cls':
            self.model = MLP(input_size=num_features, hidden_sizes=self.hidden_sizes, output_size=1,
                             activation='sigmoid')
            criterion = nn.BCELoss(reduction='none')
            val_criterion = nn.BCELoss()
        elif self.task == 'multi_cls':
            self.model = MLP(input_size=num_features, hidden_sizes=self.hidden_sizes, output_size=self.n_classes,
                             activation=None)
            criterion = nn.CrossEntropyLoss(reduction='none')
            val_criterion = nn.CrossEntropyLoss()
        elif self.task == 'regression':
            self.model = MLP(input_size=num_features, hidden_sizes=self.hidden_sizes, output_size=1,
                             activation=None)
            criterion = nn.MSELoss(reduction='none')
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

        best_train_sample_acc = 0
        best_val_sample_acc = 0
        best_test_sample_acc = 0
        best_train_acc = 0
        best_val_acc = 0
        best_test_acc = 0
        best_train_rmse = np.inf
        best_val_rmse = np.inf
        best_test_rmse = np.inf
        best_train_sample_rmse = np.inf
        best_val_sample_rmse = np.inf
        best_test_sample_rmse = np.inf
        train_idx = train_dataset.data1_idx.detach().cpu().numpy()
        train_labels = train_dataset.data1_labels.detach().cpu().numpy()
        if self.task in ['binary_cls', 'multi_cls']:
            train_labels = train_labels.astype(np.int)

        if y_scaler is None:
            answer_all = dict(zip(train_idx, train_labels))
        else:
            answer_all = dict(zip(train_idx, y_scaler.inverse_transform(train_labels.reshape(-1, 1)).flatten()))
        print("Start training")
        summary(self.model, torch.zeros([self.train_batch_size, num_features]).to(self.device))
        print(str(self))
        for epoch in range(self.num_epochs):
            # train
            train_loss = 0.0
            train_sample_correct = 0
            train_total_samples = 0
            n_train_batches = 0
            train_sample_mse = 0.0
            train_pred_all = {}
            self.model.train()
            for data, labels, weights, idx1, idx2 in tqdm(train_loader, desc="Train"):
                weights = weights.to(self.device).float()
                data = data.to(self.device).float()
                labels = labels.to(self.device).float()
                sim_scores = data[:, 0]
                data = data[:, 1:]
                optimizer.zero_grad()

                outputs = self.model(data)
                if self.task == 'binary_cls':
                    outputs = outputs.flatten()
                    losses = criterion(outputs, labels)
                    preds = outputs > 0.5
                elif self.task == 'multi_cls':
                    losses = criterion(outputs, labels.long())
                    preds = torch.argmax(outputs, dim=1)
                elif self.task == 'regression':
                    outputs = outputs.flatten()
                    losses = criterion(outputs, labels)
                    preds = outputs
                else:
                    assert False, "Unsupported task"
                n_correct = torch.count_nonzero(preds == labels).item()

                if self.merge_mode == 'avg':
                    sim_weights = weights
                    loss = torch.mean(losses * weights)
                else:
                    assert False
                loss.backward()

                optimizer.step()

                train_loss += torch.mean(losses).item()
                train_sample_correct += n_correct
                train_total_samples += data.shape[0]
                n_train_batches += 1
                if self.task == 'regression' and y_scaler is not None:
                    outputs = y_scaler.inverse_transform(
                        outputs.reshape(-1, 1).detach().cpu().numpy()).flatten()
                    labels = y_scaler.inverse_transform(
                        labels.reshape(-1, 1).detach().cpu().numpy()).flatten()
                    outputs = torch.from_numpy(outputs)
                    labels = torch.from_numpy(labels)
                train_sample_mse = train_sample_mse / (train_total_samples + data.shape[0]) * train_total_samples + \
                                   torch.sum((outputs - labels) ** 2).item() / (train_total_samples + data.shape[0])

                # calculate final prediction
                sim_weights = sim_weights.detach().cpu().numpy().flatten()
                idx1 = idx1.detach().cpu().numpy()
                idx2 = idx2.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                for i1, i2, score, out in zip(idx1, idx2, sim_weights, outputs):
                    if i1 in train_pred_all:
                        train_pred_all[i1].append((out, score, i2))
                    else:
                        train_pred_all[i1] = [(out, score, i2)]

            train_loss /= n_train_batches
            train_sample_acc = train_sample_correct / train_total_samples
            train_sample_rmse = np.sqrt(train_sample_mse)

            train_correct = 0
            train_error = []
            for i, pred_all in train_pred_all.items():
                pred = self.merge_pred(pred_all, i)
                train_error.append((answer_all[i] - pred) ** 2)
                # noinspection PyUnboundLocalVariable
                if answer_all[i] == pred:
                    train_correct += 1
            train_rmse = np.sqrt(np.average(train_error))

            assert len(answer_all) == len(train_pred_all)
            train_acc = train_correct / len(train_pred_all)

            # validation and test
            val_loss, val_sample_acc, val_acc, val_sample_rmse, val_rmse = \
                self.eval_merge_score(val_dataset, val_criterion, 'Val', y_scaler=y_scaler)
            test_loss, test_sample_acc, test_acc, test_sample_rmse, test_rmse = \
                self.eval_merge_score(test_dataset, val_criterion, 'Test', y_scaler=y_scaler)
            if self.use_scheduler:
                scheduler.step(val_loss)

            if val_acc > best_val_acc or val_rmse < best_val_rmse:
                best_train_acc = train_acc
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_train_sample_acc = train_sample_acc
                best_val_sample_acc = val_sample_acc
                best_test_sample_acc = test_sample_acc
                best_train_rmse = train_rmse
                best_val_rmse = val_rmse
                best_test_rmse = test_rmse
                best_train_sample_rmse = train_sample_rmse
                best_val_sample_rmse = val_sample_rmse
                best_test_sample_rmse = test_sample_rmse

                if self.model_save_path is not None:
                    torch.save(self.model.state_dict(), self.model_save_path)

            if self.task in ['binary_cls', 'multi_cls']:
                print("Epoch {}: {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}\n"
                      "          {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}\n"
                      "          {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}\n"
                      "          {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}\n"
                      "          {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}\n"
                      .format(epoch + 1, "Loss:", train_loss, val_loss, test_loss,
                              "Sample Acc:", train_sample_acc, val_sample_acc, test_sample_acc,
                              "Best Sample Acc:", best_train_sample_acc, best_val_sample_acc, best_test_sample_acc,
                              "Acc:", train_acc, val_acc, test_acc,
                              "Best Acc", best_train_acc, best_val_acc, best_test_acc))

                self.writer.add_scalars('Loss', {'Train': train_loss,
                                                 'Validation': val_loss,
                                                 'Test': test_loss}, epoch + 1)
                self.writer.add_scalars('Sample Accuracy', {'Train': train_sample_acc,
                                                            'Validation': val_sample_acc,
                                                            'Test': test_sample_acc}, epoch + 1)
                self.writer.add_scalars('Accuracy', {'Train': train_acc,
                                                     'Validation': val_acc,
                                                     'Test': test_acc}, epoch + 1)
            elif self.task == 'regression':
                print("Epoch {}: {:<18s}: Train {:.4f}, Val {:.4f}, Test {:.4f}\n"
                      "          {:<18s}: Train {:.4f}, Val {:.4f}, Test {:.4f}\n"
                      "          {:<18s}: Train {:.4f}, Val {:.4f}, Test {:.4f}\n"
                      "          {:<18s}: Train {:.4f}, Val {:.4f}, Test {:.4f}\n"
                      "          {:<18s}: Train {:.4f}, Val {:.4f}, Test {:.4f}\n"
                      .format(epoch + 1, "Loss:", train_loss, val_loss, test_loss,
                              "Sample RMSE:", train_sample_rmse, val_sample_rmse, test_sample_rmse,
                              "Best Sample RMSE:", best_train_sample_rmse, best_val_sample_rmse, best_test_sample_rmse,
                              "RMSE:", train_rmse, val_rmse, test_rmse,
                              "Best RMSE:", best_train_rmse, best_val_rmse, best_test_rmse))

                self.writer.add_scalars('Loss', {'Train': train_loss,
                                                 'Validation': val_loss,
                                                 'Test': test_loss}, epoch + 1)
                self.writer.add_scalars('Sample RMSE', {'Train': train_sample_rmse,
                                                        'Validation': val_sample_rmse,
                                                        'Test': test_sample_rmse}, epoch + 1)
                self.writer.add_scalars('RMSE', {'Train': train_rmse,
                                                 'Validation': val_rmse,
                                                 'Test': test_rmse}, epoch + 1)
            else:
                assert False, "Unsupported task"

        if self.task in ['binary_cls', 'multi_cls']:
            return best_train_sample_acc, best_val_sample_acc, best_test_sample_acc, \
                   best_train_acc, best_val_acc, best_test_acc
        elif self.task == 'regression':
            return best_train_rmse, best_val_rmse, best_test_rmse, \
                   best_train_rmse, best_val_rmse, best_test_rmse
        else:
            assert False

    def eval_merge_score(self, val_dataset, loss_criterion=None, name='Val', y_scaler=None):
        assert self.model is not None, "Model has not been initialized"

        val_loader = DataLoader(val_dataset, batch_size=self.test_batch_size, shuffle=False,
                                num_workers=self.num_workers, multiprocessing_context=self.multiprocess_context)

        val_idx = val_dataset.data1_idx.detach().cpu().numpy()
        val_labels = val_dataset.data1_labels.detach().cpu().numpy()
        if self.task in ['binary_cls', 'multi_cls']:
            val_labels = val_labels.astype(np.int)

        if y_scaler is None:
            answer_all = dict(zip(val_idx, val_labels))
        else:
            answer_all = dict(zip(val_idx, y_scaler.inverse_transform(val_labels.reshape(-1, 1)).flatten()))
        val_pred_all = {}

        val_loss = 0.0
        val_sample_correct = 0
        val_total_samples = 0
        val_sample_mse = 0.0
        n_val_batches = 0
        with torch.no_grad():
            self.model.eval()
            for data, labels, weights, idx1, idx2 in tqdm(val_loader, desc=name):

                data = data.to(self.device).float()
                labels = labels.to(self.device).float()
                sim_scores = data[:, 0]
                data = data[:, 1:]

                outputs = self.model(data)
                if self.merge_mode == 'avg':
                    sim_weights = weights
                else:
                    assert False

                if self.task == 'binary_cls':
                    outputs = outputs.flatten()
                    if loss_criterion is not None:
                        loss = loss_criterion(outputs, labels)
                        val_loss += loss.item()
                    preds = outputs > 0.5
                elif self.task == 'multi_cls':
                    if loss_criterion is not None:
                        loss = loss_criterion(outputs, labels.long())
                        val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                elif self.task == 'regression':
                    outputs = outputs.flatten()
                    if loss_criterion is not None:
                        loss = loss_criterion(outputs, labels)
                        val_loss += loss.item()
                    preds = outputs
                else:
                    assert False, "Unsupported task"
                n_correct = torch.count_nonzero(preds == labels).item()

                val_sample_correct += n_correct
                val_total_samples += data.shape[0]
                n_val_batches += 1
                if self.task == 'regression' and y_scaler is not None:
                    outputs = y_scaler.inverse_transform(
                        outputs.reshape(-1, 1).detach().cpu().numpy()).flatten()
                    labels = y_scaler.inverse_transform(
                        labels.reshape(-1, 1).detach().cpu().numpy()).flatten()
                    outputs = torch.from_numpy(outputs)
                    labels = torch.from_numpy(labels)
                val_sample_mse = val_sample_mse / (val_total_samples + data.shape[0]) * val_total_samples + \
                                 torch.sum((outputs - labels) ** 2).item() / (val_total_samples + data.shape[0])

                # calculate final predictions
                sim_weights = sim_weights.detach().cpu().numpy().flatten()
                idx1 = idx1.detach().cpu().numpy()
                idx2 = idx2.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                # noinspection PyUnboundLocalVariable
                for i1, i2, score, out in zip(idx1, idx2, sim_weights, outputs):
                    # noinspection PyUnboundLocalVariable
                    if i1 in val_pred_all:
                        val_pred_all[i1].append((out, score, i2))
                    else:
                        val_pred_all[i1] = [(out, score, i2)]

        val_sample_rmse = np.sqrt(val_sample_mse)
        val_correct = 0
        val_error = []
        for i, pred_all in val_pred_all.items():
            pred = self.merge_pred(pred_all, i)
            val_error.append((answer_all[i] - pred) ** 2)
            if answer_all[i] == pred:
                val_correct += 1
        val_acc = val_correct / len(val_pred_all)
        val_rmse = np.sqrt(np.average(val_error))

        val_sample_acc = val_sample_correct / val_total_samples
        if loss_criterion is not None:
            val_loss /= n_val_batches
        else:
            val_loss = -1.

        return val_loss, val_sample_acc, val_acc, val_sample_rmse, val_rmse

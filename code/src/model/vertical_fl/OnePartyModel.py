from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummaryX import summary

import torch_optimizer as adv_optim
from tqdm import tqdm
import deprecation
import warnings

from model.base.MLP import MLP
import metric
import metric.base


class BaseModel:
    def __init__(self, model_name="", val_rate=0.1, test_rate=0.2, device='cpu', hidden_sizes=None,
                 train_batch_size=128, test_batch_size=128, num_epochs=100, learning_rate=1e-3,
                 weight_decay=1e-4, num_workers=4, use_scheduler=False, sche_factor=0.1,
                 sche_patience=10, sche_threshold=0.0001, writer_path=None, model_save_path=None,
                 task='binary_cls', n_classes=2, metrics=None):
        self.n_classes = n_classes
        self.task = task
        self.model_name = model_name
        self.sche_threshold = sche_threshold
        self.sche_patience = sche_patience
        self.sche_factor = sche_factor
        self.use_scheduler = use_scheduler
        self.num_workers = num_workers
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            warnings.warn("GPU is not available, use CPU instead.")
            self.device = torch.device('cpu')
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.model_save_path = model_save_path
        self.test_rate = test_rate
        self.val_rate = val_rate
        self.multiprocess_context = 'fork' if self.num_workers > 0 else None
        self.metrics = metrics
        self.metrics_f = [self.str2func(m) for m in metrics]

        if hidden_sizes is None:
            self.hidden_sizes = [50, 50]
        else:
            self.hidden_sizes = hidden_sizes

        self.model = None
        if writer_path is not None:
            self.writer = SummaryWriter(log_dir=writer_path)
        else:
            self.writer = None

    def str2func(self, metric_str: str) -> metric.base.BaseMetric:
        if metric_str == 'accuracy':
            return metric.Accuracy()
        elif metric_str == 'rmse':
            return metric.RMSE()
        elif metric_str == 'r2_score':
            return metric.R2Score()
        elif metric_str == 'mae':
            return metric.MAE()
        else:
            assert False, "Unsupported metric"

    def merge_pred(self, pred_all: list):
        """
        Merge the prediction of a sample and make final prediction
        :param pred_all: List[<pred>: T, <score>:float]
        :return: <final prediction>: T
        """
        raise NotImplementedError

    @staticmethod
    def split_data(data, labels, val_rate=0.1, test_rate=0.2, seed=0):
        print("Splitting...")
        indices = np.arange(data.shape[0])
        train_val_data, test_data, train_val_labels, test_labels, train_val_idx, test_idx = \
            train_test_split(data, labels, indices, test_size=test_rate, random_state=seed)
        split_val_rate = val_rate / (1. - test_rate)
        train_data, val_data, train_labels, val_labels, train_idx, val_idx = \
            train_test_split(train_val_data, train_val_labels, train_val_idx, test_size=split_val_rate,
                             random_state=seed)
        return train_data, val_data, test_data, train_labels, val_labels, test_labels, train_idx, val_idx, test_idx

    def _train(self, train_X, val_X, test_X, train_y, val_y, test_y, train_idx=None, val_idx=None, test_idx=None,
               y_scaler=None):
        print("Loading data")
        if train_idx is None:
            train_dataset = TensorDataset(torch.tensor(train_X).float(), torch.tensor(train_y).float())
        else:  # need to calculate final accuracy
            train_dataset = TensorDataset(torch.tensor(train_X).float(), torch.tensor(train_y).float(),
                                          torch.tensor(train_idx).int())
        # IMPORTANT: Set num_workers to 0 to prevent deadlock on RTX3090 for unknown reason.
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                  num_workers=0)

        print("Prepare for training")
        if self.task == 'binary_cls':
            output_dim = 1
            model = MLP(input_size=train_X.shape[1], hidden_sizes=self.hidden_sizes, output_size=output_dim,
                        activation='sigmoid')
            criterion = nn.BCELoss()
        elif self.task == 'multi_cls':
            output_dim = self.n_classes
            model = MLP(input_size=train_X.shape[1], hidden_sizes=self.hidden_sizes, output_size=output_dim,
                        activation=None)
            criterion = nn.CrossEntropyLoss()
        elif self.task == 'regression':
            output_dim = 1
            model = MLP(input_size=train_X.shape[1], hidden_sizes=self.hidden_sizes, output_size=output_dim,
                        activation='sigmoid')
            criterion = nn.MSELoss()
        else:
            assert False, "Unsupported task"
        model = model.to(self.device)
        self.model = model
        optimizer = adv_optim.Lamb(model.parameters(),
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
        summary(self.model, next(iter(train_loader))[0].to(self.device))
        print(str(self))
        for epoch in range(self.num_epochs):
            if train_idx is not None:
                train_pred_all = {}
            # train
            train_loss = 0.0
            n_train_batches = 0
            model.train()
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

                outputs = model(data)
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
                train_loss += loss.item()

                if train_idx is not None:  # calculate final predictions
                    sim_scores = data[:, 0].detach().cpu().numpy()
                    # noinspection PyUnboundLocalVariable
                    idx = idx.detach().cpu().numpy()
                    # noinspection PyUnboundLocalVariable
                    for i, score, pred in zip(idx, sim_scores, preds):
                        # noinspection PyUnboundLocalVariable
                        if i in train_pred_all:
                            train_pred_all[i].append((pred, score))
                        else:
                            train_pred_all[i] = [(pred, score)]

            train_loss /= n_train_batches

            if train_idx is not None:
                all_preds = []
                all_labels = []
                for i, pred_all in train_pred_all.items():
                    pred = self.merge_pred(pred_all)
                    all_preds.append(pred)
                    # noinspection PyUnboundLocalVariable
                    all_labels.append(answer_all[i])
                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)

            train_metric_scores = []
            for metric_f in self.metrics_f:
                train_metric_scores.append(metric_f(all_preds, all_labels))

            # validation and test
            val_loss, val_metric_scores = self.eval_score(val_X, val_y, criterion, val_idx,
                                                          'Val', y_scaler=y_scaler)
            test_loss, test_metric_scores = self.eval_score(test_X, test_y, criterion, test_idx,
                                                            'Test', y_scaler=y_scaler)
            if self.use_scheduler:
                scheduler.step(val_loss)

            # The first metric determines early stopping
            if self.metrics[0] in ['accuracy', 'r2_score']:
                is_best = (val_metric_scores[0] > best_val_metric_scores[0])
            elif self.metrics[0] in ['rmse', 'mae']:
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

        return best_test_metric_scores

    @deprecation.deprecated()
    def __eval_score(self, val_X, val_y, loss_criterion, val_idx=None, name='Val', y_scaler=None):
        assert self.model is not None, "Model has not been initialized"
        if val_idx is None:
            val_dataset = TensorDataset(torch.tensor(val_X).float(), torch.tensor(val_y).float())
        else:
            val_dataset = TensorDataset(torch.tensor(val_X).float(), torch.tensor(val_y).float(),
                                        torch.tensor(val_idx).int())
        val_loader = DataLoader(val_dataset, batch_size=self.test_batch_size, shuffle=False,
                                num_workers=self.num_workers, multiprocessing_context=self.multiprocess_context)

        val_loss = 0.0
        val_sample_correct = 0
        val_mse = 0.0
        val_total_samples = 0
        n_val_batches = 0
        if val_idx is not None:
            answer_all = dict(zip(val_idx, val_y))
            val_pred_all = {}
        with torch.no_grad():
            self.model.eval()
            for info in tqdm(val_loader, desc=name):
                if val_idx is not None:
                    data, labels, idx = info
                else:
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
                    if y_scaler is not None:
                        preds = y_scaler.inverse_transform(
                            preds.reshape(-1, 1).detach().cpu().numpy()).flatten()
                        labels = y_scaler.inverse_transform(
                            labels.reshape(-1, 1).detach().cpu().numpy()).flatten()
                        preds = torch.from_numpy(preds)
                        labels = torch.from_numpy(labels)
                else:
                    assert False, "Unsupported task"
                n_correct = torch.count_nonzero(preds == labels).item()

                val_sample_correct += n_correct
                val_mse = val_mse / (val_total_samples + data.shape[0]) * val_total_samples + \
                          torch.sum((preds - labels) ** 2).item() / (val_total_samples + data.shape[0])
                val_total_samples += data.shape[0]
                n_val_batches += 1

                if val_idx is not None:  # calculate final predictions
                    sim_scores = data[:, 0].detach().cpu().numpy()
                    idx = idx.detach().cpu().numpy()
                    preds = preds.detach().cpu().numpy()
                    # noinspection PyUnboundLocalVariable
                    for i, score, pred in zip(idx, sim_scores, preds):
                        # noinspection PyUnboundLocalVariable
                        if i in val_pred_all:
                            val_pred_all[i].append((pred, score))
                        else:
                            val_pred_all[i] = [(pred, score)]

        val_acc = -1.
        if val_idx is not None:
            val_correct = 0
            for i, pred_all in val_pred_all.items():
                pred = self.merge_pred(pred_all)
                # noinspection PyUnboundLocalVariable
                if answer_all[i] == pred:
                    val_correct += 1
            val_acc = val_correct / len(val_pred_all)

        val_sample_acc = val_sample_correct / val_total_samples
        if loss_criterion is not None:
            val_loss /= n_val_batches
        else:
            val_loss = -1.

        val_rmse = np.sqrt(val_mse)

        return val_loss, val_sample_acc, val_acc, val_rmse

    def eval_score(self, val_X, val_y, loss_criterion=None, val_idx=None, name='Val', y_scaler=None):
        assert self.model is not None, "Model has not been initialized"
        if val_idx is None:
            val_dataset = TensorDataset(torch.tensor(val_X).float(), torch.tensor(val_y).float())
        else:
            val_dataset = TensorDataset(torch.tensor(val_X).float(), torch.tensor(val_y).float(),
                                        torch.tensor(val_idx).int())

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
        if val_idx is not None:
            answer_all = dict(zip(val_idx, val_y))
            val_pred_all = {}
        with torch.no_grad():
            self.model.eval()
            for info in tqdm(val_loader, desc=name):
                if val_idx is not None:
                    data, labels, idx = info
                else:
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

                if val_idx is not None:  # calculate final predictions
                    sim_scores = data[:, 0].detach().cpu().numpy()
                    idx = idx.detach().cpu().numpy()
                    # noinspection PyUnboundLocalVariable
                    for i, score, pred in zip(idx, sim_scores, preds):
                        # noinspection PyUnboundLocalVariable
                        if i in val_pred_all:
                            val_pred_all[i].append((pred, score))
                        else:
                            val_pred_all[i] = [(pred, score)]

        if val_idx is not None:
            all_preds = []
            all_labels = []
            for i, pred_all in val_pred_all.items():
                pred = self.merge_pred(pred_all)
                all_preds.append(pred)
                # noinspection PyUnboundLocalVariable
                all_labels.append(answer_all[i])
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

        if loss_criterion is not None:
            val_loss /= n_val_batches
        else:
            val_loss = -1.

        metric_scores = []
        for metric_f in self.metrics_f:
            metric_scores.append(metric_f(all_preds, all_labels))

        return val_loss, metric_scores

    @deprecation.deprecated()
    def predict_sample(self, val_X):
        assert self.model is not None, "Model has not been initialized"
        val_dataset = TensorDataset(torch.tensor(val_X).float())
        val_loader = DataLoader(val_dataset, batch_size=self.test_batch_size, shuffle=False,
                                num_workers=self.num_workers, multiprocessing_context=self.multiprocess_context)

        pred_y = torch.zeros(0, val_X.shape[1])
        with torch.no_grad():
            self.model.eval()
            for data in tqdm(val_loader, desc="Val"):
                data = data.to(self.device)
                outputs = self.model(data).flatten()
                pred = outputs > 0.5
                pred_y = torch.cat([pred_y, pred], dim=0)

        return pred_y.detach().cpu().numpy()

    def __str__(self):
        attrs = vars(self)
        output = '\n'.join("%s=%s" % item for item in attrs.items())
        return output


class OnePartyModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def merge_pred(self, pred_all: list):
        return pred_all[0][0]

    def train_all(self, data, labels):
        start_time = datetime.now()
        train_X, val_X, test_X, train_y, val_y, test_y, _, _, _ = \
            self.split_data(data, labels, val_rate=0.1, test_rate=0.2)

        result = self._train(train_X, val_X, test_X, train_y, val_y, test_y,
                           np.arange(train_X.shape[0]), np.arange(val_X.shape[0]), np.arange(test_X.shape[0]))
        time_duration_sec = (datetime.now() - start_time).seconds
        print("Training time (sec): {}".format(time_duration_sec))
        return result


    def train_single(self, data, labels, scale=True):
        train_X, val_X, test_X, train_y, val_y, test_y, _, _, _ = \
            self.split_data(data, labels, val_rate=0.1, test_rate=0.2)
        y_scaler = None
        if scale:
            x_scaler = StandardScaler()
            train_X = x_scaler.fit_transform(train_X)
            val_X = x_scaler.transform(val_X)
            test_X = x_scaler.transform(test_X)
            y_scaler = MinMaxScaler(feature_range=(0, 1))
            train_y = y_scaler.fit_transform(train_y.reshape(-1, 1)).flatten()
            val_y = y_scaler.transform(val_y.reshape(-1, 1)).flatten()
            test_y = y_scaler.transform(test_y.reshape(-1, 1)).flatten()

        start_time = datetime.now()
        result = self._train(train_X, val_X, test_X, train_y, val_y, test_y, y_scaler=y_scaler)
        time_duration_sec = (datetime.now() - start_time).seconds
        print("Training time (sec): {}".format(time_duration_sec))
        return result

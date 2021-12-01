import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from nltk.metrics.distance import edit_distance
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
# import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset

import torch_optimizer as optim
from tqdm import tqdm
from torchsummaryX import summary

from model.vertical_fl.MergeSimModel import MergeSimModel
from model.base import DLRM, MLP
from utils import get_split_points


class GameMergeDataset(Dataset):
    def __init__(self, data, labels, data_idx):
        assert data.shape[0] == data_idx.shape[0] == labels.shape[0]

        data1_order = np.argsort(data_idx[:, 0])
        self.data = data[data1_order]
        self.labels = labels[data1_order]
        self.data_idx = data_idx[:, 0][data1_order]
        self.data_idx_split_points = get_split_points(self.data_idx, self.data_idx.shape[0])
        self.data1_idx = self.data_idx[self.data_idx_split_points[:-1]]

        # print("Grouping data")
        # grouped_labels = {}
        # grouped_data = {}
        # for i in range(data_idx.shape[0]):
        #     idx1, idx2 = data_idx[i]
        #     new_data2 = np.concatenate([idx2.reshape(1, 1), data[i].reshape(1, -1)], axis=1)
        #     if idx1 in grouped_data:
        #         grouped_data[idx1] = np.concatenate([grouped_data[idx1], new_data2], axis=0)
        #     else:
        #         grouped_data[idx1] = new_data2
        #     grouped_labels[idx1] = labels[i]
        # print("Done")
        #
        # group_labels_idx = np.array(list(grouped_labels.keys()))
        # group_labels = np.array(list(grouped_labels.values()))
        # group_data_idx = np.array(list(grouped_data.keys()))
        # group_data = np.array(list(grouped_data.values()), dtype='object')
        #
        # print("Sorting data")
        # group1_order = group_labels_idx.argsort()
        # group2_order = group_data_idx.argsort()
        #
        # group_labels_idx = group_labels_idx[group1_order]
        # group_labels = group_labels[group1_order]
        # group_data_idx = group_data_idx[group2_order]
        # group_data = group_data[group2_order]
        # assert (group_labels_idx == group_data_idx).all()
        # print("Done")
        #
        # self.data1_idx: np.ndarray = group_labels_idx
        # self.labels: torch.Tensor = torch.from_numpy(group_labels)
        #
        # print("Retrieve data")
        # weight_list = []
        # data_idx_list = []
        # self.data_idx_split_points = [0]
        # for i in range(self.data1_idx.shape[0]):
        #     d2_size = group_data[i].shape[0]  # remove index
        #
        #     weight = torch.ones(d2_size) / d2_size
        #     weight_list.append(weight)
        #
        #     idx = torch.from_numpy(np.repeat(self.data1_idx[i].item(), d2_size, axis=0))
        #     data_idx_list.append(idx)
        #     self.data_idx_split_points.append(self.data_idx_split_points[-1] + idx.shape[0])
        # print("Done")
        #
        # self.data = torch.from_numpy(np.concatenate(group_data, axis=0)[:, 1:])  # remove index
        # self.weights = torch.cat(weight_list, dim=0)
        # self.data_idx = torch.cat(data_idx_list, dim=0)

    def __len__(self):
        return self.data1_idx.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start = self.data_idx_split_points[idx]
        end = self.data_idx_split_points[idx + 1]

        return self.data[start:end], self.labels[start], self.data_idx[start:end], self.data1_idx[idx]


class GameMergeSimModel(MergeSimModel):
    def __init__(self, num_common_features, top_mlp_units=None, dense_mlp_units=None, emb_dim=16, **kwargs):
        super().__init__(num_common_features, **kwargs)
        self.emb_dim = emb_dim
        self.dense_mlp_units = dense_mlp_units
        self.top_mlp_units = top_mlp_units

    @staticmethod
    def fillna_col(trn_df, vld_df, tst_df, column: str, fill):
        if fill == 'mode':
            trn_df[column] = trn_df[column].fillna(trn_df[column].mode()[0])
            vld_df[column] = vld_df[column].fillna(trn_df[column].mode()[0])
            tst_df[column] = tst_df[column].fillna(trn_df[column].mode()[0])
        elif fill == 'mean':
            trn_df[column] = trn_df[column].fillna(trn_df[column].mean())
            vld_df[column] = vld_df[column].fillna(trn_df[column].mean())
            tst_df[column] = tst_df[column].fillna(trn_df[column].mean())
        elif type(fill) is int or type(fill) is float:
            trn_df[column] = trn_df[column].fillna(fill)
            vld_df[column] = vld_df[column].fillna(fill)
            tst_df[column] = tst_df[column].fillna(fill)
        else:
            assert False

    @staticmethod
    def std_scale_col(trn_df, vld_df, tst_df, column):
        scaler = StandardScaler()
        trn_df[[column]] = scaler.fit_transform(trn_df[[column]])
        vld_df[[column]] = scaler.transform(vld_df[[column]])
        tst_df[[column]] = scaler.transform(tst_df[[column]])

    @staticmethod
    def similarity_score(a: str, b: str):
        return 1 - edit_distance(a, b) / max(len(a), len(b))

    def cal_sim_scores_head(self, arr1, arr2, head=3):
        """
        Calculate similarity scores for the strings which share the same head characters
        :param arr1: numpy string array
        :param arr2: numpy string array
        :param head: number of head characters.
        :return: Dict{Tuple(<str1 in arr1>, <str2 in arr2>): <sim_score>, ...}
        """

        arr1 = np.unique(arr1)
        arr2 = np.unique(arr2)
        print("Unique arr1 strings: {}, unique arr2 strings: {}".format(len(arr1), len(arr2)))

        # blocking
        block_dict_steam = {}
        for record in arr1:
            key = record[:head]
            if key in block_dict_steam:
                block_dict_steam[key].append(record)
            else:
                block_dict_steam[key] = [record]

        block_dict_ign = {}
        for record in arr2:
            key = record[:head]
            if key in block_dict_ign:
                block_dict_ign[key].append(record)
            else:
                block_dict_ign[key] = [record]

        print("#blocks in arr1: {}".format(len(block_dict_steam)))
        print("#blocks in arr2: {}".format(len(block_dict_ign)))

        # Compare
        title_sim_scores = {}
        for s_key, s_block in tqdm(block_dict_steam.items()):
            if s_key not in block_dict_ign:
                continue

            i_block = block_dict_ign[s_key]
            for s_title in s_block:
                for i_title in i_block:
                    sim_score = self.similarity_score(s_title, i_title)
                    title_sim_scores[(s_title, i_title)] = sim_score

            # print("Block {} matched".format(s_key))
        print("Got {} pairs of matched strings".format(len(title_sim_scores)))
        return title_sim_scores

    def match(self, steam_data, ign_data, labels=None, sim_scores=None, **kwargs):
        if sim_scores is None:
            print("Calculate sim scores")
            sim_scores = self.cal_sim_scores_head(steam_data['title'], ign_data['title'])

        ign_map = {}
        cnt = 0
        for row in ign_data.to_numpy():
            game = row[0]
            assert game not in ign_map
            ign_map[game] = cnt
            cnt += 1

        steam_map = {}
        cnt = 0
        for row in steam_data.to_numpy():
            game = row[0]
            if game not in steam_map:
                steam_map[game] = int(row[1])
            # else:
            #     assert steam_map[game] == int(row[1])
            cnt += 1

        # align on titles
        align_info = []
        num_pairs = len(sim_scores)
        print("Start aligning, got {} pairs".format(num_pairs))
        for (s_title, i_title), score in sim_scores.items():
            if s_title in steam_map:
                align_info.append([steam_map[s_title], ign_map[i_title], score])
            else:
                num_pairs -= 1
        align_info_df = pd.DataFrame(align_info, columns=['appid', 'ign_index', 'sim_score'])
        print("Aligning finished. {} pairs remained.".format(num_pairs))

        # merge ign records
        print("Merging ign records")
        ign_data['ign_index'] = range(ign_data.shape[0])
        ign_align = pd.merge(align_info_df, ign_data, on='ign_index', sort=False)
        # ign_align = ign_align.drop(columns=['ign_index'])
        ign_align.rename({'title': 'ign_title'}, axis=1, inplace=True)
        print("Finished merging with ign records, got {} lines".format(len(ign_align.index)))

        # merge steam records
        print("Merging steam records")
        steam_data['steam_index'] = range(steam_data.shape[0])
        steam_data.rename({'title': 'steam_title'}, axis=1, inplace=True)
        two_party_data_df = pd.merge(steam_data, ign_align, how='left', on='appid', sort=True)
        print("Finished merging with steam records, got {} lines".format(len(two_party_data_df.index)))

        return two_party_data_df

    @staticmethod
    def var_collate_fn(batch):
        data = torch.from_numpy(np.concatenate([item[0] for item in batch], axis=0))
        labels = torch.from_numpy(np.stack([item[1] for item in batch]))
        idx = torch.from_numpy(np.concatenate([item[2] for item in batch], axis=0))
        idx_unique = np.array([item[3] for item in batch], dtype=np.int)
        return data, labels, idx, idx_unique

    def train_dlrm(self, steam_train_data, steam_val_data, steam_test_data, ign_data, sim_score_cache_path=None,
                   data_cache_path=None):
        if data_cache_path is not None and os.path.isfile(data_cache_path):
            print("Loading datasets from cache")
            with open(data_cache_path, 'rb') as f:
                trn_set, vld_set, tst_set, counts = pickle.load(f)
            print("Loaded.")
        else:
            if sim_score_cache_path is None or not os.path.isfile(sim_score_cache_path):
                steam_titles = np.unique(np.concatenate([
                    np.unique(steam_train_data['title'].to_numpy()),
                    np.unique(steam_train_data['title'].to_numpy()),
                    np.unique(steam_train_data['title'].to_numpy()),
                ]))
                ign_titles = np.unique(ign_data['title'].to_numpy())
                sim_scores = self.cal_sim_scores_head(steam_titles, ign_titles)
                with open(sim_score_cache_path, 'wb') as f:
                    pickle.dump(sim_scores, f)
            else:
                with open(sim_score_cache_path, 'rb') as f:
                    sim_scores = pickle.load(f)

            train_data_df = self.match(steam_train_data, ign_data, sim_scores=sim_scores)
            val_data_df = self.match(steam_val_data, ign_data, sim_scores=sim_scores)
            test_data_df = self.match(steam_test_data, ign_data, sim_scores=sim_scores)

            trn_df = train_data_df.drop(columns=['ign_title', 'steam_title'])
            vld_df = val_data_df.drop(columns=['ign_title', 'steam_title'])
            tst_df = test_data_df.drop(columns=['ign_title', 'steam_title'])

            print("Normalize features and fill N/A")
            self.std_scale_col(trn_df, vld_df, tst_df, 'price')
            cols = ['appid', 'steamid', 'type', 'release_year', 'required_age', 'is_multiplayer']

            self.fillna_col(trn_df, vld_df, tst_df, 'score_phrase', fill='mode')
            self.fillna_col(trn_df, vld_df, tst_df, 'editors_choice', fill='mode')
            self.fillna_col(trn_df, vld_df, tst_df, 'score', fill='mean')
            self.fillna_col(trn_df, vld_df, tst_df, 'sim_score', fill=0)
            self.std_scale_col(trn_df, vld_df, tst_df, 'score')
            self.fillna_col(trn_df, vld_df, tst_df, 'genre', fill='mode')
            cols += ['score_phrase', 'genre', 'editors_choice']

            # category to int
            print("Transform categorical features to integer")
            counts = []
            cat2i_steamid = None
            cat2i_appid = None
            for col in cols:
                cats = sorted(trn_df[col].unique().tolist())
                cat2i = {cat: i for i, cat in enumerate(cats)}
                counts.append(len(cat2i))
                trn_df[col] = trn_df[col].transform(lambda cat: cat2i[cat])
                vld_df[col] = vld_df[col].transform(lambda cat: cat2i[cat])
                tst_df[col] = tst_df[col].transform(lambda cat: cat2i[cat])
                if col == 'appid':
                    cat2i_appid = cat2i
                elif col == 'steamid':
                    cat2i_steamid = cat2i
                print("Column {} finished".format(col))

            print("Counts: {}".format(counts))

            print("Loading steam keys and labels")
            steam_vld_df = steam_val_data
            steam_tst_df = steam_test_data
            steam_vld_df['appid'] = steam_vld_df['appid'].transform(lambda cat: cat2i_appid[cat])
            steam_tst_df['appid'] = steam_tst_df['appid'].transform(lambda cat: cat2i_appid[cat])
            steam_vld_df['steamid'] = steam_vld_df['steamid'].transform(lambda cat: cat2i_steamid[cat])
            steam_tst_df['steamid'] = steam_tst_df['steamid'].transform(lambda cat: cat2i_steamid[cat])
            steam_vld_keys = [tuple(x) for x in steam_vld_df[['appid', 'steamid']].values.tolist()]
            steam_vld_label = dict(zip(steam_vld_keys, steam_vld_df['label'].values.tolist()))
            steam_tst_keys = [tuple(x) for x in steam_tst_df[['appid', 'steamid']].values.tolist()]
            steam_tst_label = dict(zip(steam_tst_keys, steam_tst_df['label'].values.tolist()))
            print("Finished. Got {} keys & labels for validation, {} keys & labels for test"
                  .format(len(steam_vld_label), len(steam_tst_label)))  # exactly the same as exact-aligned samples

            print("Saving steam keys and labels to cache")
            with open("cache/game_sim_steam_vld_label.pkl", 'wb') as vld_file, \
                    open("cache/game_sim_steam_tst_label.pkl", 'wb') as tst_file, \
                    open("cache/game_sim_cat2i_steamid.pkl", 'wb') as cat2i_steamid_file, \
                    open("cache/game_sim_cat2i_appid.pkl", 'wb') as cat2i_appid_file:
                pickle.dump(steam_vld_label, vld_file)
                pickle.dump(steam_tst_label, tst_file)
                pickle.dump(cat2i_appid, cat2i_appid_file)
                pickle.dump(cat2i_steamid, cat2i_steamid_file)

            print("Prepare features for training")
            x_trn = [trn_df.appid, trn_df.steamid, trn_df.type, trn_df.release_year, trn_df.required_age,
                     trn_df.is_multiplayer, trn_df.score_phrase, trn_df.genre, trn_df.editors_choice,
                     trn_df[['price']].astype('float32'), trn_df[['score']].astype('float32'),
                     trn_df[['sim_score']].astype('float32')]
            x_vld = [vld_df.appid, vld_df.steamid, vld_df.type, vld_df.release_year, vld_df.required_age,
                     vld_df.is_multiplayer, vld_df.score_phrase, vld_df.genre, vld_df.editors_choice,
                     vld_df[['price']].astype('float32'), vld_df[['score']].astype('float32'),
                     vld_df[['sim_score']].astype('float32')]
            x_tst = [tst_df.appid, tst_df.steamid, tst_df.type, tst_df.release_year, tst_df.required_age,
                     tst_df.is_multiplayer, tst_df.score_phrase, tst_df.genre, tst_df.editors_choice,
                     tst_df[['price']].astype('float32'), tst_df[['score']].astype('float32'),
                     tst_df[['sim_score']].astype('float32')]

            x_trn = np.concatenate([col.to_numpy().reshape(-1, 1) for col in x_trn], axis=1)
            x_vld = np.concatenate([col.to_numpy().reshape(-1, 1) for col in x_vld], axis=1)
            x_tst = np.concatenate([col.to_numpy().reshape(-1, 1) for col in x_tst], axis=1)
            y_trn = trn_df.label.astype('float32').to_numpy()
            y_vld = vld_df.label.astype('float32').to_numpy()
            y_tst = tst_df.label.astype('float32').to_numpy()

            trn_idx = np.vstack([trn_df['steam_index'].to_numpy(), trn_df['ign_index'].to_numpy()]).T
            vld_idx = np.vstack([vld_df['steam_index'].to_numpy(), vld_df['ign_index'].to_numpy()]).T
            tst_idx = np.vstack([tst_df['steam_index'].to_numpy(), tst_df['ign_index'].to_numpy()]).T

            trn_set = GameMergeDataset(x_trn, y_trn, trn_idx)
            vld_set = GameMergeDataset(x_vld, y_vld, vld_idx)
            tst_set = GameMergeDataset(x_tst, y_tst, tst_idx)

            print("Saving dataset and counts to cache")
            with open(data_cache_path, 'wb') as f:
                pickle.dump([trn_set, vld_set, tst_set, counts], f)
            print("Saved")

        train_loader = DataLoader(trn_set, batch_size=self.train_batch_size, shuffle=True,
                                  num_workers=self.num_workers, multiprocessing_context=self.multiprocess_context,
                                  collate_fn=self.var_collate_fn)

        denses = [1, 1]
        self.model = DLRM(self.top_mlp_units, self.dense_mlp_units, self.emb_dim, counts, denses)
        self.sim_model = MLP(input_size=self.num_common_features, hidden_sizes=self.sim_hidden_sizes,
                             output_size=1, activation='sigmoid')

        self.model = self.model.to(self.device)
        self.sim_model = self.sim_model.to(self.device)
        criterion = torch.nn.BCELoss()
        val_criterion = nn.BCELoss()
        main_optimizer = optim.Lamb(self.model.parameters(),
                                    lr=self.learning_rate, weight_decay=self.weight_decay)
        sim_optimizer = optim.Lamb(self.sim_model.parameters(),
                                   lr=self.sim_learning_rate, weight_decay=self.sim_weight_decay)
        if self.use_scheduler:
            scheduler = ReduceLROnPlateau(main_optimizer, factor=self.sche_factor,
                                          patience=self.sche_patience,
                                          threshold=self.sche_threshold)
        else:
            scheduler = None

        output_dim = 1

        best_train_sample_acc = 0
        best_val_sample_acc = 0
        best_test_sample_acc = 0
        best_train_acc = 0
        best_val_acc = 0
        best_test_acc = 0
        print("Start training")
        summary(self.model, next(iter(train_loader))[0].to(self.device))
        summary(self.sim_model, torch.zeros([1, self.num_common_features if self.feature_wise_sim else 1])
                .to(self.device))
        print(str(self))
        for epoch in range(self.num_epochs):
            # train
            train_loss = 0.0
            train_sample_correct = 0
            train_total_samples = 0
            n_train_batches = 0
            self.sim_model.train()
            self.model.train()
            for data_batch, labels, idx1, idx1_unique in tqdm(train_loader, desc="Train Main"):
                data_batch = data_batch.to(self.device).float()
                labels = labels.to(self.device).float()

                sim_scores = data_batch[:, -1].reshape(-1, 1)
                data = data_batch[:, :-1]

                # train main model
                # self.model.train()
                # self.sim_model.eval()
                main_optimizer.zero_grad()
                outputs = self.model(data)
                if self.merge_mode in ['sim_model_avg', 'common_model_avg']:
                    sim_weights = self.sim_model(sim_scores) + 1e-7  # prevent dividing zero
                else:
                    assert False, "Unsupported merge mode"
                outputs_batch = torch.zeros([0, output_dim]).to(self.device)
                idx1_split_points = self.get_split_points(idx1, idx1.shape[0])
                labels_sim = torch.zeros(0).to(self.device)
                for i in range(idx1_unique.shape[0]):
                    start = idx1_split_points[i]
                    end = idx1_split_points[i + 1]
                    output_i = torch.sum(outputs[start:end] * sim_weights[start:end]) \
                               / torch.sum(sim_weights[start:end])
                    # bound threshold
                    output_i[output_i > 1.] = 1.
                    output_i[output_i < 0.] = 0.

                    outputs_batch = torch.cat([outputs_batch, output_i.reshape(-1, 1)], dim=0)
                    labels_sim = torch.cat([labels_sim, labels[i].repeat(end - start)], dim=0)
                if self.task == 'binary_cls':
                    outputs_batch = outputs_batch.flatten()
                    loss = criterion(outputs_batch, labels)
                    preds = outputs_batch > 0.5
                elif self.task == 'multi_cls':
                    loss = criterion(outputs_batch, labels.long())
                    preds = torch.argmax(outputs_batch, dim=1)
                else:
                    assert False, "Unsupported task"
                n_correct = torch.count_nonzero(preds == labels).item()

                if (epoch + 1) % self.update_sim_freq == 0:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                main_optimizer.step()

                # train sim model
                # self.sim_model.train()
                # self.model.eval()
                if (epoch + 1) % self.update_sim_freq == 0:
                    outputs = self.model(data)
                    if self.merge_mode in ['sim_model_avg', 'common_model_avg']:
                        sim_weights = self.sim_model(sim_scores) + 1e-7
                    else:
                        assert False, "Unsupported merge mode"
                    sim_optimizer.zero_grad()

                    outputs_batch = torch.zeros([0, output_dim]).to(self.device)
                    for i in range(idx1_unique.shape[0]):
                        start = idx1_split_points[i]
                        end = idx1_split_points[i + 1]
                        output_i = torch.sum(outputs[start:end] * sim_weights[start:end]) \
                                   / torch.sum(sim_weights[start:end])
                        # bound threshold
                        output_i[output_i > 1.] = 1.
                        output_i[output_i < 0.] = 0.

                        outputs_batch = torch.cat([outputs_batch, output_i.reshape(-1, 1)], dim=0)

                    if self.task == 'binary_cls':
                        outputs_batch = outputs_batch.flatten()
                        loss_batch = criterion(outputs_batch, labels)
                    elif self.task == 'multi_cls':
                        loss_batch = criterion(outputs_batch, labels.long())
                    else:
                        assert False, "Unsupported task"
                    loss_batch.backward()
                    sim_optimizer.step()

                train_loss += loss.item()
                train_sample_correct += n_correct
                train_total_samples += idx1_unique.shape[0]
                n_train_batches += 1

            train_loss /= n_train_batches
            train_acc = train_sample_acc = train_sample_correct / train_total_samples

            # validation and test
            val_loss, val_sample_acc, val_acc = self.eval_merge_score(vld_set, val_criterion, 'Val')
            test_loss, test_sample_acc, test_acc = self.eval_merge_score(tst_set, val_criterion, 'Test')
            if self.use_scheduler:
                scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_train_acc = train_acc
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_train_sample_acc = train_sample_acc
                best_val_sample_acc = val_sample_acc
                best_test_sample_acc = test_sample_acc
                if self.model_save_path is not None:
                    torch.save(self.model.state_dict(), self.model_save_path)
                    torch.save(self.sim_model.state_dict(), self.sim_model_save_path)

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
        return best_train_sample_acc, best_val_sample_acc, best_test_sample_acc, \
               best_train_acc, best_val_acc, best_test_acc

    def eval_merge_score(self, val_dataset, loss_criterion=None, name='Val', y_scaler=None):
        assert self.model is not None, "Model has not been initialized"

        val_loader = DataLoader(val_dataset, batch_size=self.train_batch_size, shuffle=True,
                                num_workers=self.num_workers, multiprocessing_context=self.multiprocess_context,
                                collate_fn=self.var_collate_fn)

        val_loss = 0.0
        val_sample_correct = 0
        val_total_samples = 0
        n_val_batches = 0
        with torch.no_grad():
            self.model.eval()
            self.sim_model.eval()
            for data_batch, labels, idx1, idx1_unique in tqdm(val_loader, desc=name):
                data_batch = data_batch.to(self.device).float()
                labels = labels.to(self.device).float()

                sim_scores = data_batch[:, -1].reshape(-1, 1)
                data = data_batch[:, :-1]

                outputs = self.model(data)
                if self.merge_mode in ['sim_model_avg', 'common_model_avg']:
                    sim_weights = self.sim_model(sim_scores) + 1e-7  # prevent dividing zero
                else:
                    assert False, "Unsupported merge mode"
                output_dim = 1
                outputs_batch = torch.zeros([0, output_dim]).to(self.device)
                idx1_split_points = self.get_split_points(idx1, idx1.shape[0])
                labels_sim = torch.zeros(0).to(self.device)
                for i in range(idx1_unique.shape[0]):
                    start = idx1_split_points[i]
                    end = idx1_split_points[i + 1]
                    output_i = torch.sum(outputs[start:end] * sim_weights[start:end]) \
                               / torch.sum(sim_weights[start:end])
                    # bound threshold
                    output_i[output_i > 1.] = 1.
                    output_i[output_i < 0.] = 0.

                    outputs_batch = torch.cat([outputs_batch, output_i.reshape(-1, 1)], dim=0)
                    labels_sim = torch.cat([labels_sim, labels[i].repeat(end - start)], dim=0)
                if self.task == 'binary_cls':
                    outputs_batch = outputs_batch.flatten()
                    loss = loss_criterion(outputs_batch, labels)
                    preds = outputs_batch > 0.5
                elif self.task == 'multi_cls':
                    loss = loss_criterion(outputs_batch, labels.long())
                    preds = torch.argmax(outputs_batch, dim=1)
                else:
                    assert False, "Unsupported task"
                n_correct = torch.count_nonzero(preds == labels).item()

                val_loss += loss.item()
                val_sample_correct += n_correct
                val_total_samples += idx1_unique.shape[0]
                n_val_batches += 1

            val_loss /= n_val_batches
            val_acc = val_sample_acc = val_sample_correct / val_total_samples

        return val_loss, val_sample_acc, val_acc

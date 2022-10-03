import torch
import torch.nn as nn


class DLRM(nn.Module):
    def __init__(self, top_mlp_units, dense_mlp_units, emb_dim, counts, denses):
        super().__init__()
        num_fea = len(counts) + len(denses)
        self.num_cat = len(counts)
        self.num_dense = len(denses)

        embs = [nn.Embedding(cnt, emb_dim) for cnt in counts]
        self.embs = nn.ModuleList(embs)

        dense_mlps = []
        for d in denses:
            mlp = []
            prev = d
            for unit in dense_mlp_units:
                mlp.append(nn.Linear(prev, unit))
                mlp.append(nn.LeakyReLU())
                prev = unit
            mlp.append(nn.Linear(prev, emb_dim))
            mlp.append(nn.LeakyReLU())
            dense_mlps.append(nn.Sequential(*mlp))
        self.dense_mlps = nn.ModuleList(dense_mlps)

        top_mlp = []
        # prev =
        prev = emb_dim * self.num_dense + int(num_fea * (num_fea - 1) / 2)
        for unit in top_mlp_units:
            top_mlp.append(nn.Linear(prev, unit))
            top_mlp.append(nn.LeakyReLU())
            prev = unit
        top_mlp.append(nn.Dropout(0.5))
        top_mlp.append(nn.Linear(prev, 1))
        top_mlp.append(nn.Sigmoid())
        self.top_mlp = nn.Sequential(*top_mlp)

    def forward(self, raw_inputs):
        inputs = [x for x in raw_inputs.T]

        cat_embs = []
        dense_embs = []

        for i in range(self.num_cat):
            emb = self.embs[i](inputs[i].long())
            # emb = self.cat_mlps[i](emb)
            cat_embs.append(emb)

        for i in range(self.num_dense):
            emb = self.dense_mlps[i](inputs[self.num_cat + i].reshape(-1, 1).float())
            dense_embs.append(emb)

        # out = torch.cat(cat_embs + dense_embs, dim=1)
        out = self.interact_features(dense_embs, cat_embs)
        out = self.top_mlp(out)
        out = torch.flatten(out)

        return out

    def interact_features(self, x, ly):
        # concatenate dense and sparse features
        (batch_size, d) = x[0].shape
        T = torch.cat(x + ly, dim=1).view((batch_size, -1, d))

        # perform a dot product
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        _, ni, nj = Z.shape
        offset = 0
        li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
        lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
        Zflat = Z[:, li, lj]

        # concatenate dense features and interactions
        R = torch.cat(x + [Zflat], dim=1)
        return R

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.05, 0.05)
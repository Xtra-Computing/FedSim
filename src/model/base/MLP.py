import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes: list, output_size=1, activation=None):
        super(MLP, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        if len(hidden_sizes) != 0:
            self.fc_layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
            for i in range(len(hidden_sizes) - 1):
                self.fc_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.fc_layers.append(nn.Linear(hidden_sizes[-1], output_size))
        else:
            self.fc_layers = nn.ModuleList([nn.Linear(input_size, output_size)])

    def forward(self, X):
        if len(list(self.fc_layers)) == 0:
            return X

        if len((list(self.fc_layers))) == 1:
            out = X
        else:
            out = F.relu(self.fc_layers[0](X))

        for fc in self.fc_layers[1:-1]:
            out = F.relu(fc(out))

        if self.activation == 'sigmoid':
            out = torch.sigmoid(self.fc_layers[-1](out))
        elif self.activation == 'tanh':
            out = torch.tanh(self.fc_layers[-1](out))
        elif self.activation == 'relu':
            out = torch.relu(self.fc_layers[-1](out))
        elif self.activation is None:
            out = self.fc_layers[-1](out)
        else:
            assert False
        return out


class DropoutInputMLP(MLP):
    def __init__(self, dropout_rate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        out = self.dropout(X)
        return super().forward(out)

import numpy as np
import os
import wget


def load_miniboone(path, download=True):
    if download and not os.path.isfile(path):
        print("Downloading MiniBooNE dataset")
        wget.download("https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt",
                      out=path)

    with open(path, 'r') as f:
        line = next(f)
        n_signal_events, n_background_events = [int(x) for x in line.split()]

    data = np.loadtxt(path, skiprows=1)
    label_signal = np.ones([n_signal_events, 1])
    label_background = np.zeros([n_background_events, 1])
    labels = np.concatenate([label_signal, label_background], axis=0)
    assert labels.shape[0] == data.shape[0]
    data_labels = np.concatenate([data, labels], axis=1)
    random_state = np.random.RandomState(0)
    random_state.shuffle(data_labels)
    return data_labels[:, :-1], data_labels[:, -1].flatten()


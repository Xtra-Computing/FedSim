# FedSim
FedSim is a **coupled vertical federated learning framework** that boosts the training with record similarities.


## Requirements
1. Install conda 4.14 following https://www.anaconda.com/products/distribution
2. Clone this repo by
```bash
git clone https://github.com/JerryLife/FedSim.git
```
3. Create environment (named `fedsim`) and install required basic modules.
```bash
conda env create -f environment.yml
conda activate fedsim
```
4. Install `torch` and `torchvision` according to your cuda version with `pip`. For RTX 3090, we installed `torch==1.8.2` and `torchvision==0.9.2` as below.
```bash
pip3 install torch==1.8.2 torchvision==0.9.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
``` 
5. Ensure all the required folders are created (which should exist upon git clone).
```bash
mkdir -p runs ckp log cache
```
## Datasets
In this repo, due to the size limit, we include two datasets `house` and `game` in the `data/` folder.
```
data
├── beijing 				(house)
│   ├── airbnb_clean.csv	(Secondary)
│   └── house_clean.csv		(Primary)
└── hdb						(hdb)
    ├── hdb_clean.csv		(Primary)
    └── school_clean.csv	(Secondary)
```
## Linkage and Training
The linkage and training of each dataset is combined in a single script.
### FedSim without adding noise
The scripts without adding noise are located under `src/` in the format of `src/train_<dataset>_<algorithm>.py`. You can run each script by

```bash
python src/train_<dataset>_<algorithm>.py [-g gpu_index] [-p perturbed_noise_on_similarity] [-k number_of_neighbors] [--mlp-merge] [-ds] [-dw]
```
> -g/--gpu: GPU index to run this script. If GPU of this index is not available, CPU will be used instead.
> -k/--top-k: Number of neighbors to extract from possible matches, which should be less than the value of "knn_k". ($K$ in the paper)
> -p/--leak-p: The probability of leakage of bloom filters. ($\tau$ in the paper)
> --mlp-merge: whether to replace CNN merge model with MLP merge model
> -ds/--disable-sort: whether to distable the sort gate
> -dw/--disable-weight: whether to disable the weight gate

Taking house dataset dataset as an example:
```bash
python src/train_beijing_fedsim.py -g 1 -p 1e0 -k 5 -ds
```
runs FedSim on house dataset with $\tau=1$ (no added noise), $K=5$, merging with CNN, disabling sort gate, enabling weight gate.

### FedSim with noise added
The scripts with adding noise are located in `src/priv_scripts` in the same format as the scripts without noise. The only difference are some hyperparamter settings. You may run these scripts by similar command. For example,
```bash
python src/train_beijing_fedsim.py -g 1 -p 1e-2 -k 5 -ds
```
runs FedSim on house dataset with noise satisfying $\tau=0.01$ added, $K=5$, merging with CNN, disabling sort gate, enabling weight gate.

## Citation
> Wu, Zhaomin, Qinbin, Li, Bingsheng, He, "A Coupled Design of Exploiting Record Similarity for Practical Vertical Federated Learning." Accepted in _Advances in Neural information processing systems_ (2022).


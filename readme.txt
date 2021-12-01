###############################################################################
##                     README of Supplementary Materials                     ##
###############################################################################

The structure of this folder is as follows.

Submission 8849 Supplementary Materials
|-- code
|   |-- data
|   |   `-- beijing
|   |       |-- airbnb_clean.csv
|   |       `-- house_clean.csv
|   |-- environment.yml
|   `-- src
|       |-- align
|       |-- metric
|       |-- model
|       |   |-- base
|       |   |-- game
|       |   |-- __init__.py
|       |   `-- vertical_fl
|       |-- preprocess
|       |   |-- beijing
|       |   |-- game
|       |   |-- hdb
|       |   |-- __init__.py
|       |   |-- ml_dataset
|       |   |-- nytaxi
|       |   |-- sklearn
|       |   `-- song
|       |-- train_*_*.py
|       `-- utils
`-- readme.txt



"Appendix.pdf" includes proofs, additional experiments, experimental details, and discussions.

"code/" includes the codes to reproduce our experiments and one example dataset (house dataset).

"readme.txt" (this file) includes the instructions to run the codes.


--------------------- Instruction of Reproduction -----------------------------

1. Name of datasets and algorithms in the code

For convenience, we adopt slightly different names of datasets and algorithms in our implementation. Therefore, the maps between the names are listed below.

	Dataset Map
Code 	  	Paper
'beijing'	'house',
'hdb' 		'hdb',
'song'		'song',
'ny' 		'taxi',
'game'	 	'game',
'syn'		'sklearn',
'frog' 		'frog',
'boone' 	'boone'

	Algorithm Map
Code 			Paper
'A'				'Solo'
'fedsim'		'FedSim'
'avgsim'		'AvgSim'
'featuresim'	'FeatureSim'
'top1sim'		'Top1Sim'
'exact'			'Exact'



2. Environment

Our implementation is based on Python 3.8.8 and Anaconda 4.9.2. The environment can be auto-configured by running

$ cd code
$ conda env create -f environment.yml
$ conda activate fedsim

Hardware requirements: GPU is not essential to run the codes, but there has to be sufficient memory to perform linkage.

Before running scripts, you have to generate required directories by

$ mkdir cache ckp log runs



3. Run the programs

All the scripts are listed under "src/train_<dataset>_<algorithm>.py". You can run each script by 

$ python src/train_<dataset>_<algorithm>.py [-g gpu_index] [-p perturbed_noise_on_similarity] [-s noise_scale_on_synthetic_datasets] [-k number_of_neighbors]

-g --gpu 	GPU index to run this script. If GPU of this index is not available, CPU will be used instead.

-k --top-k 	Number of neighbors to extract from possible matches, which should be less than the value of "knn_k". This parameter is only implemented on real-world datasets.

-p --leak-p	The probability of leakage of bloom filters. (\tau in the paper)

-s --noise-scale	Noise on the identifiers of synthetic datasets (\sigma_{cf} in the paper), which is only applicable for synthetic datasets.

Taking house dataset and sklearn dataset as an example:

$ python src/train_beijing_fedsim.py -g 1 -p 1e0 -k 5
$ python src/train_syn_fedsim.py -g 0 -p 1e-3 -s 0.1



4. Reproduction of Experiments

For experiments in Section 6.2, you can simply run each script of real-world datasets without changing parameters. But with synthetic datasets, you should change parameter "-s" in [0.0, 0.1, 0.2]. 

For experiments in Section 6.3, you need to change parameter "-p" (\tau) to different values.

For experiments in Appendix E, you need to change the parameter "-k" (K) to get the results. 



****************************** Known Bug **************************************

On RTX3090, when running FedSim from linkage to training, it might throw a Cuda exception at the start of training. To work around this issue, simply rerun the program, which will auto-load from the cache and run normally.
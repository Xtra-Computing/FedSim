import recordlinkage.datasets
import os
import sys



if __name__ == '__main__':

    os.chdir(sys.path[0] + "/../../../data/krebs")  # change working directory
    os.environ["RL_DATA"] = os.getcwd()     # set path of recordinglinkage dataset

    data = recordlinkage.datasets.load_krebsregister(block=1)

    pass
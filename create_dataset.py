import os, sys, yaml, pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from utils.dataset_custom import *

import random
random.seed(49297)



def main():
    #####################################################
    cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
    output_path = cfg["output_path"]
    dataset_output_path = cfg["dataset_output_path"]
    if not os.path.isdir(dataset_output_path):
        os.mkdir(dataset_output_path)
    #####################################################
    episode_index = pd.read_csv(os.path.join(output_path, "episodes_index.csv"))
    episode_datas = [(row["SUBJECT_ID"], row["EPISODE"], row["TIMESERIES_RAW"]) for i,row in episode_index.iterrows()]
    create_mortality(dataset_output_path, episode_datas, 24)
    create_mortality(dataset_output_path, episode_datas, 48)


if __name__ == "__main__":
    main()
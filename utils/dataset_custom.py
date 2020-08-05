import os, yaml, shutil
from tqdm import tqdm
import pandas as pd
import random
random.seed(49297)



def create_mortality(output_dataset_path, episode_datas, hours = 24):
    output_path = output_dataset_path
    output_dataset_path = os.path.join(output_dataset_path,str(hours))
    if not os.path.isdir(output_dataset_path):
        os.mkdir(output_dataset_path)

    eps = 1e-6
    episodes_df = pd.DataFrame(columns=("SUBJECT_ID", "LABEL", "LOS", "EPISODE_PATH", "TIMESERIES_PATH"))

    for episode_data in tqdm(episode_datas, desc='Iterating over patients'):
        subject_id, episode_index_path, episode_timeseries_path = episode_data
        ##########################################################
        if not (os.path.exists(episode_timeseries_path) and (os.path.exists(episode_index_path))):
            print("\n\tno episode in ICU of ", subject_id)
            continue
        episode_timeseries = pd.read_csv(episode_timeseries_path)
        if episode_timeseries.shape[0] == 0:
            print("\n\tno events in ICU of ", subject_id)
            continue
        episode_timeseries = episode_timeseries[((episode_timeseries.HOURS > 0) & (episode_timeseries.HOURS < hours+eps))]
        ##########################################################
        episode = pd.read_csv(episode_index_path)
        if episode.shape[0]==0:
            print("\n\tno episode in ICU of ", subject_id)
            continue
        mortality = int(episode.iloc[0]["MORTALITY_INHOSPITAL"])
        los = 24.0 * episode.iloc[0]['Length of Stay']  # in hours
        if pd.isnull(los):
            print("\n\tlength of stay is missing of ", subject_id)
            continue
        # only keep admissions with record length > 24/48 hrs.
        if los < hours + eps:
            continue
        ##########################################################
        output_dataset_path_i  = os.path.join(output_dataset_path,str(subject_id))
        if not os.path.isdir(output_dataset_path_i):
            os.mkdir(output_dataset_path_i)
        output_episode_timeseries_path = os.path.join(output_dataset_path_i, episode_timeseries_path.split("/")[-1])
        episode_timeseries.to_csv(output_episode_timeseries_path)
        output_episode_path = os.path.join(output_dataset_path_i, episode_index_path.split("/")[-1])
        shutil.copy(episode_index_path, output_episode_path)
        ##########################################################
        s = pd.Series({ "SUBJECT_ID": subject_id, "LABEL": mortality, "LOS": los, \
                        "EPISODE_PATH":output_episode_path, "TIMESERIES_PATH":output_episode_timeseries_path})
        episodes_df = episodes_df.append(s, ignore_index=True)


    path_index = os.path.join(output_path, "episodes_dataset.csv")
    episodes_df.to_csv(path_index)
    print("Number of created samples:", episodes_df.shape[0])
    print("Saving index at:", path_index)




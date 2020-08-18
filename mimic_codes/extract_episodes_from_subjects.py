import os, sys, yaml
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from utils.preprocessing import *
from utils.read_csv import *
from utils.episode_custom import *

def multitask(idx, subject_dirs, output_subject_path, itemids2variables, icds2variables):
    subject_dir = subject_dirs[idx]
    dn = os.path.join(output_subject_path, str(subject_dir))
    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except:
        sys.stderr.write('No dir for subject: {}\n'.format(subject_id))
        return None
    #############################################################
    # print("reading tables...")
    # read tables of this subject
    stays = read_stays(dn)
    # TODO: what to use
    diagnoses = read_diagnoses(dn, icds2variables)
    procedures = read_procedures(dn)
    # map itemids to variables in event
    events = read_events(dn, itemids2variables)
    #############################################################
    # print("reading static data...")
    episodic_data = assemble_episodic_data(stays, diagnoses, procedures)
    # cleaning and converting to time series
    # events = clean_events(events)
    if events.shape[0] == 0:
        # no valid events for this subject
        sys.stderr.write('No valid events for this subject: {}\n'.format(subject_id))
        return None
    # print("reading timeseries data...")
    timeseries_processed, timeseries_raw = convert_events_to_timeseries(events, variables=itemids2variables.values() \
                                                                        , itemids=itemids2variables.keys())

    # extracting separate episodes
    for i in range(stays.shape[0]):
        stay_id = stays.ICUSTAY_ID.iloc[i]
        intime = stays.INTIME.iloc[i]
        outtime = stays.OUTTIME.iloc[i]

        episode_processed = get_events_for_stay(timeseries_processed, stay_id, intime, outtime)
        episode_raw = get_events_for_stay(timeseries_raw, stay_id, intime, outtime)

        if episode_processed.shape[0] == 0 or episode_raw.shape[0] == 0:
            # no data for this episode
            return None

        episode = add_hours_elpased_to_events(episode_processed, intime).set_index('HOURS').sort_index(axis=0)
        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "HOURS" else x))
        episode = episode[columns_sorted]
        episode_path = os.path.join(dn, 'episode{}_timeseries_processed.csv'.format(i + 1))
        episode.to_csv(episode_path, index_label='HOURS')

        episode2 = add_hours_elpased_to_events(episode_raw, intime).set_index('HOURS').sort_index(axis=0)
        columns = list(episode2.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if str(x) == "HOURS" else str(x)))
        episode2 = episode2[columns_sorted]
        episode_path2 = os.path.join(dn, 'episode{}_timeseries_raw.csv'.format(i + 1))
        episode2.to_csv(episode_path2, index_label='HOURS')

        episodic_data_path = os.path.join(dn, 'episode{}.csv'.format(i + 1))
        episodic_data.loc[episodic_data.index == stay_id].to_csv(episodic_data_path, index_label='Icustay')

    return [subject_id, episode_path, episode_path2, episodic_data_path]

def main():
    cfg = yaml.load(open("./config.yaml","r"), Loader=yaml.FullLoader)
    output_path = cfg["output_path"]
    output_subject_path = cfg["output_subject_path"]
    variable_map = cfg["variable_map"]
    #################################################################
    # read all itemids2variables map in events and prescriptions
    itemids2variables = read_variables(variable_map)
    icds2variables = read_variables_icd(variable_map)
    #################################################################
    # save index for episodes of all subjects
    episodes_df = pd.DataFrame(columns=("SUBJECT_ID","EPISODE","TIMESERIES_RAW","TIMESERIES_PROCESSED"))
    episodes_subjectids = []
    episodic_data_index = []
    episode_index = []
    episode_raw_index = []
    ls = os.listdir(output_subject_path)

    print("starting multitask...")
    pool = Pool(10)
    partial_process = partial(multitask, subject_dirs= ls, output_subject_path = output_subject_path, \
                              itemids2variables = itemids2variables, icds2variables = icds2variables)

    N = len(ls)
    res = pool.map(partial_process, range(N))
    pool.close()
    pool.join()
    print("finishing multitask...")


    for i in res:
        if i:
            s = pd.Series({ "SUBJECT_ID": i[0], "EPISODE": i[3], "TIMESERIES_RAW":i[2], "TIMESERIES_PROCESSED":i[1]})
            episodes_df = episodes_df.append(s, ignore_index=True)
    episodes_df.to_csv(os.path.join(output_path,"episodes_index.csv"))
    print("# of samples:" , episodes_df.shape[0])

def main2():
    cfg = yaml.load(open("./config.yaml","r"), Loader=yaml.FullLoader)
    output_path = cfg["output_path"]
    output_subject_path = cfg["output_subject_path"]
    variable_map = cfg["variable_map"]
    #################################################################
    # read all itemids2variables map in events and prescriptions
    itemids2variables = read_variables(variable_map)
    icds2variables = read_variables_icd(variable_map)
    #################################################################
    # save index for episodes of all subjects
    episodes_df = pd.DataFrame(columns=("SUBJECT_ID","EPISODE","TIMESERIES_RAW","TIMESERIES_PROCESSED"))
    episodes_subjectids = []
    episodic_data_index = []
    episode_index = []
    episode_raw_index = []
    #################################################################
    # split episodes
    for subject_dir in tqdm(os.listdir(output_subject_path), desc='Iterating over subjects'):
        dn = os.path.join(output_subject_path, str(subject_dir))
        try:
            subject_id = int(subject_dir)
            if not os.path.isdir(dn):
                raise Exception
        except:
            sys.stderr.write('No dir for subject: {}\n'.format(subject_id))
            continue
        #############################################################
        #print("reading tables...")
        # read tables of this subject
        stays = read_stays(dn)
        # TODO: what to use
        diagnoses = read_diagnoses(dn, icds2variables)
        procedures = read_procedures(dn)
        # map itemids to variables in event
        events = read_events(dn, itemids2variables)
        #############################################################
        #print("reading static data...")
        episodic_data = assemble_episodic_data(stays, diagnoses, procedures)
        # cleaning and converting to time series
        #events = clean_events(events)
        if events.shape[0] == 0:
            # no valid events for this subject
            sys.stderr.write('No valid events for this subject: {}\n'.format(subject_id))
            continue
        #print("reading timeseries data...")
        timeseries_processed, timeseries_raw = convert_events_to_timeseries(events, variables=itemids2variables.values() \
                                                                        , itemids = itemids2variables.keys())

        # extracting separate episodes
        for i in range(stays.shape[0]):
            stay_id = stays.ICUSTAY_ID.iloc[i]
            intime = stays.INTIME.iloc[i]
            outtime = stays.OUTTIME.iloc[i]

            episode_processed = get_events_for_stay(timeseries_processed, stay_id, intime, outtime)
            episode_raw = get_events_for_stay(timeseries_raw, stay_id, intime, outtime)

            if episode_processed.shape[0] == 0 or episode_raw.shape[0] == 0:
                # no data for this episode
                continue

            episode = add_hours_elpased_to_events(episode_processed, intime).set_index('HOURS').sort_index(axis=0)
            columns = list(episode.columns)
            columns_sorted = sorted(columns, key=(lambda x: "" if x == "HOURS" else x))
            episode = episode[columns_sorted]
            episode_path = os.path.join(dn, 'episode{}_timeseries_processed.csv'.format(i+1))
            episode.to_csv(episode_path,index_label='HOURS')
            episode_index.append(episode_path)

            episode2 = add_hours_elpased_to_events(episode_raw, intime).set_index('HOURS').sort_index(axis=0)
            columns = list(episode2.columns)
            columns_sorted = sorted(columns, key=(lambda x: "" if str(x) == "HOURS" else str(x)))
            episode2 = episode2[columns_sorted]
            episode_path = os.path.join(dn, 'episode{}_timeseries_raw.csv'.format(i+1))
            episode2.to_csv(episode_path,index_label='HOURS')
            episode_raw_index.append(episode_path)

            episodic_data_path = os.path.join(dn,'episode{}.csv'.format(i+1))
            episodic_data.loc[episodic_data.index == stay_id].to_csv(episodic_data_path,index_label='Icustay')
            episodic_data_index.append(episodic_data_path)

            episodes_subjectids.append(subject_id)
    for i in range(len(episodes_subjectids)):
        s = pd.Series({ "SUBJECT_ID": episodes_subjectids[i], "EPISODE": episodic_data_index[i], \
                          "TIMESERIES_RAW":episode_raw_index[i], "TIMESERIES_PROCESSED":episode_index[i]})
        episodes_df = episodes_df.append(s, ignore_index=True)
    episodes_df.to_csv(os.path.join(output_path,"episodes_index.csv"))


if __name__ == "__main__":
    main()

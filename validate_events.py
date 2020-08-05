import os, yaml
import numpy as np
import pandas as pd
from tqdm import tqdm



def main():
    cfg = yaml.load(open("./config.yaml","r"), Loader=yaml.FullLoader)
    output_path = cfg["output_path"]
    subjects_root_path = cfg["output_subject_path"]
    tables = cfg["event_tables"]
    tables.append("prescriptions")
    n_events = 0                   # total number of events
    empty_hadm = 0                 # HADM_ID is empty in events.csv. We exclude such events.
    no_hadm_in_stay = 0            # HADM_ID does not appear in stays.csv. We exclude such events.
    no_icustay = 0                 # ICUSTAY_ID is empty in events.csv. We try to fix such events.
    recovered = 0                  # empty ICUSTAY_IDs are recovered according to stays.csv files (given HADM_ID)
    could_not_recover = 0          # empty ICUSTAY_IDs that are not recovered. This should be zero.
    icustay_missing_in_stays = 0   # ICUSTAY_ID does not appear in stays.csv. We exclude such events.
    
    subdirectories = os.listdir(subjects_root_path)
    subjects = list(filter(str.isdigit, subdirectories))

    for subject in tqdm(subjects, desc='Iterating over subjects'):
        # read stays.csv
        stays_df = pd.read_csv(os.path.join(subjects_root_path, subject, 'stays.csv'), index_col=False,
                               dtype={'HADM_ID': str, "ICUSTAY_ID": str})
        stays_df.columns = stays_df.columns.str.upper()

        # assert that there is no row with empty ICUSTAY_ID or HADM_ID
        assert(not stays_df['ICUSTAY_ID'].isnull().any())
        assert(not stays_df['HADM_ID'].isnull().any())

        # assert there are no repetitions of ICUSTAY_ID or HADM_ID
        # since admissions with multiple ICU stays were excluded
        assert(len(stays_df['ICUSTAY_ID'].unique()) == len(stays_df['ICUSTAY_ID']))
        assert(len(stays_df['HADM_ID'].unique()) == len(stays_df['HADM_ID']))


        for table in tables:
            event_path = os.path.join(subjects_root_path, subject, table.lower()+".csv")
            if not os.path.exists(event_path):
                continue
            events_df = pd.read_csv(event_path, index_col=False,
                                dtype={'HADM_ID': str, "ICUSTAY_ID": str})
            events_df.columns = events_df.columns.str.upper()
            if table in ["prescriptions"]:
                events_df = events_df.rename(columns = {"VALUENUM":"VALUE"})
            n_events += events_df.shape[0]

            # drop all events whose HADM_ID is empty
            # TODO: maybe we can recover HADM_ID by looking at ICUSTAY_ID
            empty_hadm += events_df['HADM_ID'].isnull().sum()
            events_df = events_df.dropna(subset=['HADM_ID'])

            merged_df = events_df.merge(stays_df, left_on=['HADM_ID'], right_on=['HADM_ID'],
                                    how='left', suffixes=['', '_r'], indicator=True)

            # drop all events whose HADM_ID is not listed in stays.csv
            # since there is no way to know the targets of that stay (for example mortality)
            no_hadm_in_stay += (merged_df['_merge'] == 'left_only').sum()
            merged_df = merged_df[merged_df['_merge'] == 'both']

            # if ICUSTAY_ID is empty in stays.csv, we try to recover it
            # we exclude all events for which we could not recover ICUSTAY_ID
            cur_no_icustay = merged_df['ICUSTAY_ID'].isnull().sum()
            no_icustay += cur_no_icustay
            merged_df.loc[:, 'ICUSTAY_ID'] = merged_df['ICUSTAY_ID'].fillna(merged_df['ICUSTAY_ID_r'])
            recovered += cur_no_icustay - merged_df['ICUSTAY_ID'].isnull().sum()
            could_not_recover += merged_df['ICUSTAY_ID'].isnull().sum()
            merged_df = merged_df.dropna(subset=['ICUSTAY_ID'])

            # now we take a look at the case when ICUSTAY_ID is present in events.csv, but not in stays.csv
            # this mean that ICUSTAY_ID in events.csv is not the same as that of stays.csv for the same HADM_ID
            # we drop all such events
            icustay_missing_in_stays += (merged_df['ICUSTAY_ID'] != merged_df['ICUSTAY_ID_r']).sum()
            merged_df = merged_df[(merged_df['ICUSTAY_ID'] == merged_df['ICUSTAY_ID_r'])]

            # filter events with nan VALUE
            filter_df = merged_df[merged_df.VALUE.notnull()]

            # write to csv
            to_write = filter_df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE']]
            to_write.to_csv(os.path.join(subjects_root_path, subject, table.lower()+"_verified.csv"), index=False)

    assert(could_not_recover == 0)
    print('n_events: {}'.format(n_events))
    print('empty_hadm: {}'.format(empty_hadm))
    print('no_hadm_in_stay: {}'.format(no_hadm_in_stay))
    print('no_icustay: {}'.format(no_icustay))
    print('recovered: {}'.format(recovered))
    print('could_not_recover: {}'.format(could_not_recover))
    print('icustay_missing_in_stays: {}'.format(icustay_missing_in_stays))


if __name__ == "__main__":
    main()

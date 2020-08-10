"""
In this step, we generate the matrix of time series for each admission.

    Discard admissoins without admittime.
    Collect records from inputevents, outputevents, chartevents, labevents, microbiologyevents and prescriptionevents.
    For possible conflictions(different values of the same feature occur at the same time):
        For numerical values:
            For inputevents/outputevents/prescriptions, we use the sum of all conflicting values.
            For labevents/chartevents/microbiologyevents, we use the mean of all conflicting values.
        For categorical values: we use the value appear first and record that confliction event in the log.
        For ratio values: we separate the ratio to two numbers and use the mean of each of them.
"""
import os
import yaml
import numpy as np
import pandas as pd
from utils.admit_split import *
from multiprocessing import Pool, cpu_count

def main():
    #####################################################
    cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
    mimic3_path = cfg["mimic3_path"]
    output_path = cfg["output_path"]
    output_admit_path = cfg["output_admit_path"]
    res_path = cfg["res_path"]
    event_tables = cfg["event_tables"]
    if not os.path.isdir(mimic3_path):
        os.mkdir(mimic3_path)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(output_admit_path):
        os.mkdir(output_admit_path)
    #####################################################
    #admits = read_admissions_table_all(mimic3_path)
    _adm = np.load(os.path.join(res_path, 'admission_ids.npy'), allow_pickle=True).tolist()
    admission_ids = _adm['admission_ids']
    read_admits_and_add_age_los_mortality_icd9(mimic3_path,output_admit_path, admission_ids)
    event_tables.append("prescriptions")
    for table in event_tables:
        read_events_table_and_break_up_by_subject_keep_all_items(output_path, table, output_admit_path, admits_to_keep=admission_ids)
    for table in event_tables:
        read_events_table_and_generate_matrix_of_time_series(res_path, table, output_admit_path, admits_to_keep=admission_ids)





if __name__ == "__main__":
    main()



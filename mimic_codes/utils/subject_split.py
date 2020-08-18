import os, csv, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas import DataFrame, Series


#################################################
# split icustays by subject
#################################################

def break_up_stays_by_subject(stays, output_path, subjects=None):
    subjects = stays.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up stays by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        stays[stays.SUBJECT_ID == subject_id].sort_values(by='INTIME').to_csv(os.path.join(dn, 'stays.csv'),
                                                                              index=False)


#################################################
# split diagnoses icd by subject
#################################################

def add_hcup_ccs_2015_groups(diagnoses, definitions):
    def_map = {}
    for dx in definitions:
        for code in definitions[dx]['codes']:
            def_map[code] = (dx, definitions[dx]['use_in_benchmark'])
    diagnoses['HCUP_CCS_2015'] = diagnoses.ICD9_CODE.apply(lambda c: def_map[c][0] if c in def_map else None)
    diagnoses['USE_IN_BENCHMARK'] = diagnoses.ICD9_CODE.apply(lambda c: int(def_map[c][1]) if c in def_map else None)
    return diagnoses

def make_phenotype_label_matrix(phenotypes, stays=None):
    phenotypes = phenotypes[['ICUSTAY_ID', 'HCUP_CCS_2015']].loc[phenotypes.USE_IN_BENCHMARK > 0].drop_duplicates()
    phenotypes['VALUE'] = 1
    phenotypes = phenotypes.pivot(index='ICUSTAY_ID', columns='HCUP_CCS_2015', values='VALUE')
    if stays is not None:
        phenotypes = phenotypes.reindex(stays.ICUSTAY_ID.sort_values())
    return phenotypes.fillna(0).astype(int).sort_index(axis=0).sort_index(axis=1)


def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None):
    subjects = diagnoses.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up diagnoses by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        diagnoses[diagnoses.SUBJECT_ID == subject_id].sort_values(by=['ICUSTAY_ID', 'SEQ_NUM'])\
                                                     .to_csv(os.path.join(dn, 'diagnoses.csv'), index=False)



#################################################
# split procedures icd by subject
#################################################
def break_up_procedures_by_subject(procedures, output_path, subjects=None):
    subjects = procedures.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up procedures by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        procedures[procedures.SUBJECT_ID == subject_id].sort_values(by=['ICUSTAY_ID', 'SEQ_NUM']) \
            .to_csv(os.path.join(dn, 'procedures.csv'), index=False)
        
#################################################
# split prescriptions icd by subject
#################################################
def break_up_prescriptions_by_subject(prescriptions, output_path, subjects=None):
    subjects = prescriptions.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up prescriptions by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        prescriptions[prescriptions.SUBJECT_ID == subject_id].sort_values(by=['ICUSTAY_ID', 'CHARTTIME']) \
            .to_csv(os.path.join(dn, 'prescriptions.csv'), index=False)


#################################################
# split events by subject
#################################################

def read_events_table_by_row(mimic3_path, table):
    #nb_rows = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219}
    reader = csv.DictReader(open(os.path.join(mimic3_path, table.upper() + '.csv'), 'r'))
    for i, row in enumerate(reader):
        if 'ICUSTAY_ID' not in row:
            row['ICUSTAY_ID'] = ''
        yield row, i


def read_events_table_and_break_up_by_subject_keep_all_items(mimic3_path, table, output_path,
                                            subjects_to_keep=None):
    nb_rows_dict = {'inputevents_cv':1483111,'inputevents_mv':1732207,'chartevents': 33866825, 'labevents': 19645103, 'outputevents': 180922}
    nb_rows = nb_rows_dict[table.lower()]
    obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE']
    if subjects_to_keep is not None:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    if subjects_to_keep is None:
        return None

    class DataStats(object):
        def __init__(self):
            self.curr_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    def write_current_observations():
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        fn = os.path.join(dn, table.lower() + '.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []

    if table.lower() in [ "inputevents_cv", "inputevents_mv", "outputevents" ]:
        discard_func = lambda row : pd.isnull(row['CHARTTIME']) or row['CHARTTIME']=='' or pd.isnull(row['VALUENUM'])\
                                    or row['VALUENUM']==''
    elif table.lower() in [ "chartevents", "labevents" ]:
        discard_func = lambda row : pd.isnull(row['CHARTTIME']) or row['CHARTTIME']=='' or pd.isnull(row['VALUENUM']) \
                                    or row['VALUENUM']=='' or pd.isnull(row['VALUEUOM']) or row['VALUEUOM']==''


    for row, row_no in tqdm(read_events_table_by_row(mimic3_path, table), desc='Processing {} table'.format(table)):

        if (row['SUBJECT_ID'] not in subjects_to_keep):
            continue
        if discard_func(row):
            continue
        ###################################################
        row_out = {'SUBJECT_ID': row['SUBJECT_ID'],
                'HADM_ID': row['HADM_ID'],
                'ICUSTAY_ID': row['ICUSTAY_ID'],
                'CHARTTIME': row['CHARTTIME'],
                'ITEMID': row['ITEMID'],
                'VALUE': row['VALUENUM']}
        ###################################################
        if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['SUBJECT_ID']:
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['SUBJECT_ID']

    if data_stats.curr_subject_id != '':
        write_current_observations()







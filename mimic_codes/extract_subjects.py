import os
import yaml
import numpy as np
import pandas as pd
from utils.read_csv import *
from utils.preprocessing import *
from utils.subject_split import *

def main():
    #####################################################
    cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
    mimic3_path = cfg["mimic3_path"]
    output_path = cfg["output_path"]
    output_subject_path = cfg["output_subject_path"]
    #phenotype_definitions = cfg["phenotype_definitions"]
    event_tables = cfg["event_tables"]
    if not os.path.isdir(mimic3_path):
        os.mkdir(mimic3_path)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    #####################################################
    # read specified columns
    patients = read_patients_table(mimic3_path)
    admits = read_admissions_table(mimic3_path)
    stays = read_icustays_table(mimic3_path)
    diagnoses = read_icd_diagnoses_table(mimic3_path)
    procedures = read_icd_procedures_table(mimic3_path)
    prescriptions = read_prescriptions_table(output_path)
    #####################################################
    # merge together
    stays = merge_on_subject_admission(stays, admits)
    stays = merge_on_subject(stays, patients)
    stays = filter_first_icustay_admission(stays)
    stays = add_length_of_admit_to_icustays(stays)
    stays = add_age_to_icustays(stays)
    stays = add_inunit_mortality_to_icustays(stays)
    stays = add_inhospital_mortality_to_icustays(stays)
    stays = add_2days_mortality_to_icustays(stays)
    stays = add_3days_mortality_to_icustays(stays)
    stays = add_30days_mortality_discharge(stays)
    stays = add_1year_mortality_discharge(stays)
    stays = filter_icustays_on_age(stays, 15)
    subjects = stays.SUBJECT_ID.unique()
    print(f"creating icustays for {len(subjects)} patients")
    stays.to_csv(os.path.join(output_path, 'all_stays.csv'), index=False)
    #####################################################
    prescriptions = filter_prescriptions_on_stays(prescriptions, stays)
    prescriptions.to_csv(os.path.join(output_path, 'all_prescriptions.csv'), index=False)
    #####################################################
    diagnoses = filter_diagnoses_on_stays(diagnoses, stays)
    diagnoses_count = count_icd_codes(diagnoses)
    #diagnoses = filter_icd_on_count(diagnoses, diagnoses_count, 0)
    diagnoses_count.to_csv(os.path.join(output_path, 'all_diagnoses_count.csv'), index=False)
    diagnoses.to_csv(os.path.join(output_path, 'all_diagnoses.csv'), index=False)
    #####################################################
    procedures = filter_procedures_on_stays(procedures, stays)
    procedures_count = count_icd_codes(procedures)
    #procedures = filter_icd_on_count(procedures, procedures_count, 0)
    procedures_count.to_csv(os.path.join(output_path, 'all_procedures_count.csv'), index=False)
    procedures.to_csv(os.path.join(output_path, 'all_procedures.csv'), index=False)
    #####################################################
    #definitions = yaml.load(open(phenotype_definitions, 'r'), Loader=yaml.FullLoader)
    """
    {'Tuberculosis': {'use_in_benchmark': False,
      'type': 'unknown',
      'id': 1,
      'codes': [
       '01000',
       '01001',
       '01002',]
        },
    }
    """
    #####################################################
    break_up_stays_by_subject(stays, output_subject_path, subjects=subjects)
    #diagnoses = add_hcup_ccs_2015_groups(diagnoses, definitions)
    break_up_diagnoses_by_subject(diagnoses, output_subject_path, subjects=subjects)
    # phenotype labels of diagnoses icd-9 code for icustays
    #phenotype_labels = make_phenotype_label_matrix(diagnoses_with_phenotypes, stays)
    #phenotype_labels.to_csv(os.path.join(output_path, 'phenotype_labels.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    #####################################################
    # procedures icd-9 code for icustays
    #break_up_procedures_by_subject(procedures, output_subject_path, subjects=subjects)
    #####################################################
    # prescriptions for icustays
    break_up_prescriptions_by_subject(prescriptions, output_subject_path, subjects=subjects)
    #####################################################
    for table in event_tables:
        read_events_table_and_break_up_by_subject_keep_all_items(output_path, table, output_subject_path, subjects_to_keep=subjects)


if __name__ == "__main__":
    main()

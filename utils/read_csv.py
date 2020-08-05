import os
import csv
import numpy as np
import pandas as pd

#################################################
# read from original csv
#################################################


def dataframe_from_csv(path, header=0, index_col=0):
    return pd.read_csv(path, header=header, index_col=index_col,low_memory=False)

def read_patients_table(mimic3_path):
    pats = dataframe_from_csv(os.path.join(mimic3_path, 'PATIENTS.csv'))
    pats = pats[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]
    pats.DOB = pd.to_datetime(pats.DOB)
    pats.DOD = pd.to_datetime(pats.DOD)
    return pats


def read_admissions_table(mimic3_path):
    admits = dataframe_from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv'))
    admits = admits[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'DIAGNOSIS', 'ADMISSION_TYPE']]
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits


def read_icustays_table(mimic3_path):
    stays = dataframe_from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv'))
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    return stays


def read_icd_diagnoses_table(mimic3_path):
    codes = dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_DIAGNOSES.csv'))
    codes = codes[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']]
    diagnoses = dataframe_from_csv(os.path.join(mimic3_path, 'DIAGNOSES_ICD.csv'))
    diagnoses = diagnoses.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
    diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']] = diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']].astype(int)
    return diagnoses

def read_icd_procedures_table(mimic3_path):
    codes = dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_PROCEDURES.csv'))
    codes = codes[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']]
    procedures = dataframe_from_csv(os.path.join(mimic3_path, 'PROCEDURES_ICD.csv'))
    procedures = procedures.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
    procedures[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']] = procedures[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']].astype(int)
    return procedures


def read_prescriptions_table(mimic3_path):
    prescriptions = dataframe_from_csv(os.path.join(mimic3_path, 'PRESCRIPTIONS.csv'))
    prescriptions.CHARTTIME = pd.to_datetime(prescriptions.CHARTTIME)
    prescriptions = prescriptions.dropna(axis=0, subset = ["VALUENUM"])
    return prescriptions

def read_prescriptions_table2(mimic3_path):
    prescriptions = dataframe_from_csv(os.path.join(mimic3_path, 'PRESCRIPTIONS.csv'))
    prescriptions.STARTDATE = pd.to_datetime(prescriptions.STARTDATE)
    prescriptions.ENDDATE = pd.to_datetime(prescriptions.ENDDATE)
    prescriptions.DOSE_UNIT_RX = prescriptions.DOSE_UNIT_RX.str.lower()
    prescriptions = prescriptions.dropna(axis=0, subset = ["DRUG","DOSE_UNIT_RX"])
    return prescriptions
















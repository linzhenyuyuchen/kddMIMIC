import re
import numpy as np
from pandas import DataFrame, Series

from .read_csv import *

#################################################
# transform value of prescriptions
#################################################

def filter_value(strs):
    strs = strs.replace(",",".")
    strs = strs.replace(". ",".")
    if len(re.findall(re.compile(r'[A-Za-z]',re.S),strs)):
        return 0
    elif len(re.findall(re.compile(r'-',re.S),strs)):
        if strs.split("-")[0] == '':
            return 0
        elif strs.split("-")[1] != '':
            return ((float(strs.split("-")[0])+float(strs.split("-")[1]))/2)
        else:
            return float(strs.split("-")[0])
    else:
        return float(strs)

def correct_value(prescriptions):
    prescriptions["DOSE_VAL_RX"] = prescriptions["DOSE_VAL_RX"].apply(filter_value)
    return prescriptions

#################################################
# remove
#################################################


def remove_icustays_with_transfers(stays):
    stays = stays[(stays.FIRST_WARDID == stays.LAST_WARDID) & (stays.FIRST_CAREUNIT == stays.LAST_CAREUNIT)]
    return stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE', 'INTIME', 'OUTTIME', 'LOS']]



#################################################
# filter
#################################################

# filter admission times between (min_times, max_times) per patient
def filter_admissions_times(admits,min_times, max_times):
    to_keep = admits.groupby("SUBJECT_ID").count()[['HADM_ID']].reset_index()
    to_keep = to_keep[(to_keep.HADM_ID >= min_times) & (to_keep.HADM_ID <= max_times)][['SUBJECT_ID']]
    admits = admits.merge(to_keep, how='inner', left_on='SUBJECT_ID', right_on='SUBJECT_ID')
    return admits

# filter icu_stay times between (min_nb_stays, max_nb_stays) per admission (filter icustay with transfer)
def filter_admissions_on_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1):
    to_keep = stays.groupby('HADM_ID').count()[['ICUSTAY_ID']].reset_index()
    to_keep = to_keep[(to_keep.ICUSTAY_ID >= min_nb_stays) & (to_keep.ICUSTAY_ID <= max_nb_stays)][['HADM_ID']]
    stays = stays.merge(to_keep, how='inner', left_on='HADM_ID', right_on='HADM_ID')
    return stays

# filter first admission per patient (SUBJECT_ID 38597 / HADM_ID 49785)
def filter_first_admission(stays):
    to_keep = stays.sort_values(by=["SUBJECT_ID","HADM_ID","ICUSTAY_ID"]).drop_duplicates(subset=["SUBJECT_ID"], keep='first')
    to_keep = to_keep[["SUBJECT_ID", "HADM_ID"]]
    stays = stays.merge(to_keep, how='inner', left_on=["SUBJECT_ID", "HADM_ID"], right_on=["SUBJECT_ID", "HADM_ID"])
    return stays

# filter first icu admission per patient (SUBJECT_ID 38597 / HADM_ID 38597)
def filter_first_icustay_admission(stays):
    to_keep = stays.sort_values(by=["SUBJECT_ID","HADM_ID","ICUSTAY_ID"]).drop_duplicates(subset=["SUBJECT_ID"], keep='first')
    to_keep = to_keep[["SUBJECT_ID","ICUSTAY_ID"]]
    stays = stays.merge(to_keep, how='inner', left_on=["SUBJECT_ID","ICUSTAY_ID"], right_on=["SUBJECT_ID","ICUSTAY_ID"])
    return stays


def filter_icustays_on_died24hour(stays):
    duration = 24 * 60 * 60
    mortality = (stays.DOD.notnull() & stays.INTIME.notnull() & (stays.INTIME <= stays.DOD) & ((stays.DOD - stays.INTIME).apply(lambda s: s / np.timedelta64(1, 's')).fillna(duration+1).astype(int) <= duration) )

    return stays[mortality]

def filter_icustays_on_age(stays, min_age=18, max_age=np.inf):
    stays = stays[(stays.AGE > min_age) & (stays.AGE <= max_age)]
    return stays


def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].drop_duplicates(), how='inner',
                           left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])

def filter_procedures_on_stays(procedures, stays):
    return procedures.merge(stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].drop_duplicates(), how='inner',
                           left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])


def filter_prescriptions_in_icu(prescriptions):
    return prescriptions[(prescriptions.CHARTTIME>= prescriptions.INTIME) & (prescriptions.CHARTTIME>= prescriptions.STARTDATE)]

def filter_prescriptions_on_stays(prescriptions, stays):
    return prescriptions.merge(stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME']].drop_duplicates(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']), how='inner',
                            left_on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'], right_on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])

def filter_prescriptions_on_count(prescriptions, threshold_count, threshold_percent):
    drug_unit = prescriptions[["DRUG","DOSE_UNIT_RX","SUBJECT_ID"]].dropna(axis=0,subset = ["DRUG","DOSE_UNIT_RX"]).sort_values(["DRUG","DOSE_UNIT_RX"])
    drug_unit = drug_unit.groupby(["DRUG","DOSE_UNIT_RX"]).count().reset_index()
    drug_unit_sum = drug_unit.groupby(["DRUG"]).sum().reset_index()
    drug_unit = drug_unit.merge(drug_unit_sum,how="inner",left_on=["DRUG"],right_on=["DRUG"])
    drug_unit["PERCENT"] = drug_unit.SUBJECT_ID_x / drug_unit.SUBJECT_ID_y
    drug_unit_filtered = drug_unit[(drug_unit.SUBJECT_ID_y>threshold_count) & (drug_unit.PERCENT>threshold_percent)]
    prescriptions = prescriptions.merge(drug_unit_filtered[["DRUG","DOSE_UNIT_RX"]].drop_duplicates(), how='inner',
                               left_on=["DRUG","DOSE_UNIT_RX"], right_on=["DRUG","DOSE_UNIT_RX"])
    return prescriptions[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE', 'ENDDATE', 'DRUG_TYPE', 'DRUG', 'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'INTIME', 'OUTTIME']]

def filter_icd_on_count(df_icd, df_count, threshold):
    df_count = df_count[df_count.COUNT > threshold][["ICD9_CODE"]]
    df_icd = df_icd.merge(df_count, how="inner", left_on=["ICD9_CODE"], right_on=["ICD9_CODE"])
    return df_icd
#################################################
# count
#################################################

def count_icd_codes(diagnoses_or_procedures):
    codes = diagnoses_or_procedures[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']].drop_duplicates().set_index('ICD9_CODE')
    codes['COUNT'] = diagnoses_or_procedures.groupby('ICD9_CODE')['ICUSTAY_ID'].count()
    codes.COUNT = codes.COUNT.fillna(0).astype(int)
    codes = codes[codes.COUNT > 0]
    return codes.sort_values('COUNT', ascending=False).reset_index()


#################################################
# merge
#################################################

def merge_on_subject(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])


def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])




#################################################
# add
#################################################

def add_age_to_icustays2(stays):
    stays['AGE'] = (stays.INTIME - stays.DOB).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24/365
    stays.loc[stays.AGE < 0, 'AGE'] = 90
    return stays

def add_age_to_icustays(stays):
    stays['AGE'] = (stays.INTIME.dt.date - stays.DOB.dt.date).apply(lambda x: int(x.days/365))
    stays.loc[stays.AGE < 0, 'AGE'] = 90
    return stays

def add_length_of_admit_to_icustays(stays):
    stays.ADMITTIME = pd.to_datetime(stays.ADMITTIME)
    stays.DISCHTIME = pd.to_datetime(stays.DISCHTIME)
    stays['LengthofAdmit'] = (stays.DISCHTIME - stays.ADMITTIME).apply(lambda s: s / np.timedelta64(1, 's'))
    return stays


def add_inhospital_mortality_to_icustays(stays):
    #mortality = stays.DOD.notnull() & ((stays.ADMITTIME <= stays.DOD) & (stays.DISCHTIME >= stays.DOD))
    #mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.ADMITTIME <= stays.DEATHTIME) & (stays.DISCHTIME >= stays.DEATHTIME)))
    mortality = stays.DEATHTIME.notnull()
    stays['MORTALITY'] = mortality.astype(int)
    stays['MORTALITY_INHOSPITAL'] = stays['MORTALITY']
    return stays


def add_inunit_mortality_to_icustays(stays):
    mortality = stays.DOD.notnull() & ((stays.INTIME <= stays.DOD) & (stays.OUTTIME >= stays.DOD))
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.INTIME <= stays.DEATHTIME) & (stays.OUTTIME >= stays.DEATHTIME)))
    stays['MORTALITY_INUNIT'] = mortality.astype(int)
    return stays

def add_2days_mortality_to_icustays(stays):
    duration = 48 * 60 * 60
    mortality = (stays.DOD.notnull() & stays.INTIME.notnull() & (stays.INTIME <= stays.DOD) & ((stays.DOD - stays.INTIME).apply(lambda s: s / np.timedelta64(1, 's')).fillna(duration+1).astype(int) <= duration) )
    stays['MORTALITY_2DAYS'] = mortality.astype(int)
    return stays

def add_3days_mortality_to_icustays(stays):
    duration = 72 * 60 * 60
    mortality = (stays.DOD.notnull() & stays.INTIME.notnull() & (stays.INTIME <= stays.DOD) & ((stays.DOD - stays.INTIME).apply(lambda s: s / np.timedelta64(1, 's')).fillna(duration+1).astype(int) <= duration) )
    stays['MORTALITY_3DAYS'] = mortality.astype(int)
    return stays

def add_30days_mortality_discharge(stays):
    duration = 30 * 24 * 60 * 60
    mortality = (stays.DOD.notnull() & stays.DISCHTIME.notnull() & (stays.DISCHTIME <= stays.DOD) & ((stays.DOD - stays.DISCHTIME).apply(lambda s: s / np.timedelta64(1, 's')).fillna(duration+1).astype(int) <= duration) )
    stays['MORTALITY_30DAYS'] = mortality.astype(int)
    return stays

def add_1year_mortality_discharge(stays):
    duration = 365 * 24 * 60 * 60
    mortality = (stays.DOD.notnull() & stays.DISCHTIME.notnull() & (stays.DISCHTIME <= stays.DOD) & ((stays.DOD - stays.DISCHTIME).apply(lambda s: s / np.timedelta64(1, 's')).fillna(duration+1).astype(int) <= duration) )
    stays['MORTALITY_1YEAR'] = mortality.astype(int)
    return stays

def add_longterm_mortality_to_icustays(stays):
    duration = 48 * 60 * 60
    mortality = (stays.DOD.notnull() & stays.INTIME.notnull() & ((stays.DOD - stays.INTIME).apply(lambda s: s / np.timedelta64(1, 's')).fillna(duration+1).astype(int) <= duration) )
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.DEATHTIME - stays.INTIME).apply(lambda s: s / np.timedelta64(1, 's')).fillna(duration+1).astype(int) <= duration) )
    stays['MORTALITY_48HOURS'] = mortality.astype(int)
    return stays

def add_shortterm_mortality_to_icustays(stays):
    duration = 24 * 60 * 60
    mortality = (stays.DOD.notnull() & ((stays.DOD - stays.INTIME).apply(lambda s: s / np.timedelta64(1, 's')).fillna(duration+1).astype(int) <= duration) )
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.DEATHTIME - stays.INTIME).apply(lambda s: s / np.timedelta64(1, 's')).fillna(duration+1).astype(int) <= duration) )
    stays['MORTALITY_24HOURS'] = mortality.astype(int)
    return stays










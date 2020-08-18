import os, re
import csv
import numpy as np
import pandas as pd

#################################################
# read from split-subject csv
#################################################

def dataframe_from_csv(path, header=0, index_col=0):
    return pd.read_csv(path, header=header, index_col=index_col,low_memory=False)


def read_stays(subject_path):
    stays = dataframe_from_csv(os.path.join(subject_path, 'stays.csv'), index_col=None)
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    stays.DOB = pd.to_datetime(stays.DOB)
    stays.DOD = pd.to_datetime(stays.DOD)
    stays.DEATHTIME = pd.to_datetime(stays.DEATHTIME)
    stays.sort_values(by=['INTIME', 'OUTTIME'], inplace=True)
    return stays


def extract_diagnosis(diagnoses, resources_path):
    to_use = dataframe_from_csv(os.path.join(resources_path, "diagnoses_icd9_to_variable_map.csv"), index_col=None)
    to_use = to_use[to_use.USE > 0]
    variables = to_use.VARIABLE.unique()
    icd_codes = to_use.ICD9_CODE.unique()

    diagnoses['VALUE'] = 1
    labels = diagnoses[['ICUSTAY_ID', 'ICD9_CODE', 'VALUE']].drop_duplicates() \
        .pivot(index='ICUSTAY_ID', columns='ICD9_CODE', values='VALUE').fillna(0).astype(int)
    for l in diagnosis_labels:
        if l not in labels:
            labels[l] = 0
    labels = labels[diagnosis_labels]
    return labels.rename(dict(zip(icd_codes, variables)), axis=1)


def extract_procedures(procedures, resources_path):
    to_use = dataframe_from_csv(os.path.join(resources_path, "procedures_icd9_to_variable_map.csv"), index_col=None)
    to_use = to_use[to_use.USE > 0]
    variables = to_use.VARIABLE.unique()
    icd_codes = to_use.ICD9_CODE.unique()

    procedures['VALUE'] = 1
    labels = procedures[['ICUSTAY_ID', 'ICD9_CODE', 'VALUE']].drop_duplicates() \
        .pivot(index='ICUSTAY_ID', columns='ICD9_CODE', values='VALUE').fillna(0).astype(int)
    for l in icd_codes:
        if l not in labels:
            labels[l] = 0
    labels = labels[icd_codes]
    return labels.rename(dict(zip(icd_codes, variables)), axis=1)


def read_diagnoses(subject_path, icds2variables):
    diagnoses = dataframe_from_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)
    diagnoses["VARIABLE"] = diagnoses["ICD9_CODE"].apply(lambda icd: icds2variables[str(int(icd))] if str(icd).isdigit() else icds2variables[str(icd)])
    diagnoses['VALUE'] = 1
    labels = diagnoses[['ICUSTAY_ID', 'VARIABLE', 'VALUE']].drop_duplicates() \
        .pivot(index='ICUSTAY_ID', columns='VARIABLE', values='VALUE').fillna(0).astype(int)
    variables = list(set(icds2variables.values()))
    for l in variables:
        if l not in labels:
            labels[l] = 0
    labels = labels[variables]
    return labels


def read_procedures(subject_path):
    #return dataframe_from_csv(os.path.join(subject_path, 'procedures.csv'), index_col=None)
    return None


def read_prescriptions(subject_path):
    return dataframe_from_csv(os.path.join(subject_path, 'prescriptions.csv'), index_col=None)

def extract_prescriptions(pres, resources_path, table):
    to_use = dataframe_from_csv(os.path.join(resources_path, table.lower()+"_to_variable_map.csv"), index_col=None)
    to_use = to_use[to_use.USE > 0][['DRUG',"VARIABLE"]]
    pres = pres.merge(to_use, left_on='DRUG', right_on='DRUG')
    del pres["DRUG"]
    origin_name = ["SUBJECT_ID","HADM_ID","ICUSTAY_ID","STARTDATE","DOSE_VAL_RX","VARIABLE"]
    to_name = ["SUBJECT_ID","HADM_ID","ICUSTAY_ID","CHARTTIME","VALUE","VARIABLE"]
    return pres.rename(dict(zip(origin_name, to_name)))

def extract_prescriptions_touse(resources_path, table):
    to_use = dataframe_from_csv(os.path.join(resources_path, table.lower()+"_to_variable_map.csv"), index_col=None)
    to_use = to_use[to_use.USE > 0][['DRUG',"VARIABLE"]]
    return to_use

def extract_events(events, resources_path, table):
    to_use = dataframe_from_csv(os.path.join(resources_path, table.lower()+"_to_variable_map.csv"), index_col=None)
    to_use = to_use[to_use.USE > 0][['ITEMID',"VARIABLE"]]
    events = events.merge(to_use, left_on='ITEMID', right_on='ITEMID')
    del events["ITEMID"]
    return events

def extract_events_touse(resources_path, table):
    to_use = dataframe_from_csv(os.path.join(resources_path, table.lower()+"_to_variable_map.csv"), index_col=None)
    to_use = to_use[to_use.USE > 0][['ITEMID',"VARIABLE"]]
    return to_use

def read_events_columns(path, remove_null):
    events = dataframe_from_csv(path, index_col=None)
    if remove_null:
        events = events[events.VALUE.notnull()]
    return events

def read_events(subject_path, itemids2variables, remove_null=True):
    input_cv_path = os.path.join(subject_path, "inputevents_cv_verified.csv")
    input_mv_path = os.path.join(subject_path, "inputevents_mv_verified.csv")
    lab_path = os.path.join(subject_path, "labevents_verified.csv")
    output_path = os.path.join(subject_path, "outputevents_verified.csv")
    chart_path = os.path.join(subject_path, "chartevents_verified.csv")
    prescriptions_path = os.path.join(subject_path, "prescriptions_verified.csv")
    ##################################################
    events = pd.DataFrame(columns=("SUBJECT_ID","HADM_ID","ICUSTAY_ID","CHARTTIME", "VALUE", "VARIABLE"))
    ##################################################
    if os.path.exists(input_cv_path):
        df = read_events_columns(input_cv_path, remove_null)
        events = pd.concat([events,df], axis=0, ignore_index=True, sort=False)
    ##################################################
    if os.path.exists(input_mv_path):
        df = read_events_columns(input_mv_path, remove_null)
        events = pd.concat([events,df], axis=0, ignore_index=True, sort=False)
    ##################################################
    if os.path.exists(lab_path):
        df = read_events_columns(lab_path, remove_null)
        events = pd.concat([events,df], axis=0, ignore_index=True, sort=False)
    ##################################################
    if os.path.exists(output_path):
        df = read_events_columns(output_path, remove_null)
        events = pd.concat([events,df], axis=0, ignore_index=True, sort=False)
    ##################################################
    if os.path.exists(chart_path):
        df = read_events_columns(chart_path, remove_null)
        events = pd.concat([events,df], axis=0, ignore_index=True, sort=False)
    ##################################################
    if os.path.exists(prescriptions_path):
        df = read_events_columns(prescriptions_path, remove_null)
        events = pd.concat([events,df], axis=0, ignore_index=True, sort=False)
    ##################################################
    events["VARIABLE"] = events["ITEMID"].apply(lambda itemid: itemids2variables[str(itemid)] if re.findall(re.compile(r'[A-Za-z]',re.S),str(itemid)) else itemids2variables[str(int(itemid))])
    events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
    events = events[~(events.VALUE < 0)]
    return events

def read_variables(resources_path):
    variables = {}
    df = dataframe_from_csv(os.path.join(resources_path, "itemids_to_variable_map.csv"), index_col=None)
    for i, row in df.iterrows():
        variables[str(row["ITEMID"])] = row["VARIABLE"]
    return variables

def read_variables2(resources_path):
    tables = ["inputevents", "labevents", "outputevents", "chartevents", "prescriptions"]
    variables = {}
    for table in tables:
        df = dataframe_from_csv(os.path.join(resources_path, table.lower()+"_to_variable_map.csv"), index_col=None)
        #to_use = to_use[to_use.USE > 0]
        for i, row in df.iterrows():
            variables[str(row["ITEMID"])] = row["VARIABLE"]
    return variables

def read_variables_icd(resources_path):
    variables = {}
    df = dataframe_from_csv(os.path.join(resources_path,"diagnoses_icd9_to_variable_map.csv"), index_col=None)
    #to_use = to_use[to_use.USE > 0]
    for i, row in df.iterrows():
        variables[str(row["ICD9_CODE"])] = row["VARIABLE"]
    return variables

def read_events2(subject_path, remove_null=True):
    events = dataframe_from_csv(os.path.join(subject_path, 'chartevents.csv'), index_col=None)
    if remove_null:
        events = events[events.VALUE.notnull()]
    events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
    events.HADM_ID = events.HADM_ID.fillna(value=-1).astype(int)
    events.ICUSTAY_ID = events.ICUSTAY_ID.fillna(value=-1).astype(int)
    events.VALUEUOM = events.VALUEUOM.fillna('').astype(str)
    # events.sort_values(by=['CHARTTIME', 'ITEMID', 'ICUSTAY_ID'], inplace=True)
    return events
#################################################
# split events by subject
# episodeN.csv <- diagnosis
# episodeN_timeseries.csv <- events
#################################################

def convert_events_to_timeseries(events, variable_column='VARIABLE', variables=[], itemids = []):
    variables = list(set(variables))
    itemids = list(set(itemids))
    metadata = events[['CHARTTIME', 'ICUSTAY_ID']].sort_values(by=['CHARTTIME', 'ICUSTAY_ID']) \
        .drop_duplicates(keep='first').set_index('CHARTTIME')
    #################################################
    """
    timeseries = events[['CHARTTIME', variable_column, 'VALUE']] \
        .fillna(0) \
        .sort_values(by=['CHARTTIME', variable_column, 'VALUE'], axis=0)
    duplcate_row = timeseries.duplicated(subset=['CHARTTIME', variable_column], keep=False)
    duplicate_data = timeseries.loc[duplcate_row,:]
    #duplicate_data_sum = duplicate_data.groupby(by=['CHARTTIME', variable_column]).agg([("VALUE",'sum')]).reset_index()
    duplicate_data_mean = duplicate_data.groupby(by=['CHARTTIME', variable_column]).agg([("VALUE",'mean')]).reset_index()
    duplicate_data_mean.columns = duplicate_data_mean.columns.droplevel(1)
    no_duplicate = timeseries.drop_duplicates(subset=['CHARTTIME', variable_column], keep=False)
    timeseries_processed = pd.concat([no_duplicate, duplicate_data_mean], sort=False)
    timeseries_processed = timeseries_processed.sort_values(by=['CHARTTIME', variable_column]) \
        .pivot(index='CHARTTIME', columns=variable_column, values='VALUE') \
        .merge(metadata, left_index=True, right_index=True) \
        .sort_index(axis=0).reset_index()
    for v in variables:
        if v not in timeseries_processed:
            timeseries_processed[v] = np.nan
    """
    #################################################
    ts = events[['CHARTTIME', 'ITEMID', 'VALUE']] \
        .sort_values(by=['CHARTTIME', 'ITEMID', 'VALUE'], axis=0) \
        .drop_duplicates(subset=['CHARTTIME', 'ITEMID'], keep='last')
    ts = ts.pivot(index='CHARTTIME', columns='ITEMID', values='VALUE') \
        .merge(metadata, left_index=True, right_index=True) \
        .sort_index(axis=0).reset_index()
    #######################
    for v in itemids:
        if v not in ts:
            ts[v] = np.nan
    #######################
    ts["systolic_blood_pressure_abp_mean"] = ts[['51', '442', '455', '6701', '220050', '220179']].apply(lambda x: x.mean(), axis=1)
    del ts["51"]; del ts["442"]; del ts["455"]; del ts["6701"]; del ts["220050"]; del ts["220179"];
    #######################
    ts["f2c"] = ts[['678', '223761']].apply(lambda x: (x.mean() - 32) * 5.0 / 9.0, axis=1)
    del ts["678"]; del ts["223761"];
    #######################
    ts["urinary_output_sum"] = ts[["40055","43175","40069","40094","40715","40473","40085","40057",\
                                   "40056","40405","40428","40086","40096","40651","226559","226560",\
                                   "226561","226584","226563","226564","226565","226567","226557",\
                                   "226558","227488","227489"]].apply(lambda x: x.sum(), axis=1)
    del ts["40055"]; del ts["43175"]; del ts["40069"]; del ts["40094"]; del ts["40715"];
    del ts["40473"]; del ts["40085"]; del ts["40057"]; del ts["40056"]; del ts["40405"];
    del ts["40428"]; del ts["40086"]; del ts["40096"]; del ts["40651"]; del ts["226559"];
    del ts["226560"]; del ts["226561"]; del ts["226584"]; del ts["226563"];
    del ts["226564"]; del ts["226565"]; del ts["226567"]; del ts["226557"];
    del ts["226558"]; del ts["227488"]; del ts["227489"];
    #######################
    ts["white_blood_cells_count_mean"] = ts[['51300', '51301']].apply(lambda x: x.mean(), axis=1)
    del ts["51300"]; del ts["51301"];
    #######################
    ts["sodium_level_mean"] = ts[['50824', '50983']].apply(lambda x: x.mean(), axis=1)
    del ts["50824"]; del ts["50983"];
    #######################
    ts["potassium_level_mean"] = ts[['50822', '50971']].apply(lambda x: x.mean(), axis=1)
    del ts["50822"]; del ts["50971"];
    #######################
    ts["tempcol"] = ts['226873'] / ts['226871']
    del ts["226873"]; del ts["226871"];
    ts["ie_ratio_mean"] = ts[['tempcol', '221']].apply(lambda x: x.mean(), axis=1)
    del ts["tempcol"]; del ts["221"];
    #######################
    ts["diastolic_blood_pressure_mean"] = ts[["8368","8440","8441","8555","220180","220051"]].apply(lambda x: x.mean(), axis=1)
    del ts["8368"]; del ts["8440"]; del ts["8441"];
    del ts["8555"]; del ts["220180"]; del ts["220051"];
    #######################
    ts["arterial_pressure_mean"] = ts[["456","52","6702","443","220052","220181","225312"]].apply(lambda x: x.mean(), axis=1)
    del ts["456"]; del ts["52"]; del ts["6702"];
    del ts["443"]; del ts["220052"]; del ts["220181"]; del ts["225312"];
    #######################
    ts["3581"] = ts['3581'] * 0.453592
    #######################
    ts["3581"] = ts['3582'] * 0.0283495
    #######################
    ts["920"] = ts['920'] * 2.54
    ts["1394"] = ts['1394'] * 2.54
    ts["4187"] = ts['4187'] * 2.54
    ts["3486"] = ts['3486'] * 2.54
    ts.columns = ts.columns.astype(str)
    #################################################
    timeseries_raw = events[['CHARTTIME', 'ITEMID', 'VALUE']] \
        .sort_values(by=['CHARTTIME', 'ITEMID', 'VALUE'], axis=0) \
        .drop_duplicates(subset=['CHARTTIME', 'ITEMID'], keep='last')
    timeseries_raw = timeseries_raw.pivot(index='CHARTTIME', columns='ITEMID', values='VALUE') \
        .merge(metadata, left_index=True, right_index=True) \
        .sort_index(axis=0).reset_index()
    for v in itemids:
        if v not in timeseries_raw:
            timeseries_raw[v] = np.nan
    timeseries_raw.columns = timeseries_raw.columns.astype(str)
    #################################################
    return ts, timeseries_raw



def get_events_for_stay(events, icustayid, intime=None, outtime=None):
    idx = (events.ICUSTAY_ID == icustayid)
    if intime is not None and outtime is not None:
        idx = idx | ((events.CHARTTIME >= intime) & (events.CHARTTIME <= outtime))
    events = events[idx]
    del events['ICUSTAY_ID']
    return events


def add_hours_elpased_to_events(events, dt, remove_charttime=True):
    events = events.copy()
    events['HOURS'] = (events.CHARTTIME - dt).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60
    if remove_charttime:
        del events['CHARTTIME']
    return events

def get_first_valid_from_timeseries(timeseries, variable):
    if variable in timeseries:
        idx = timeseries[variable].notnull()
        if idx.any():
            loc = np.where(idx)[0][0]
            return timeseries[variable].iloc[loc]
    return np.nan



#################################################
# transform gender
#################################################

g_map = {'F': 1, 'M': 2, 'OTHER': 3, '': 0}


def transform_gender(gender_series):
    global g_map
    return {'Gender': gender_series.fillna('').apply(lambda s: g_map[s] if s in g_map else g_map['OTHER'])}


#################################################
# transform ethnicity
#################################################

e_map = {'ASIAN': 1,
         'BLACK': 2,
         'CARIBBEAN ISLAND': 2,
         'HISPANIC': 3,
         'SOUTH AMERICAN': 3,
         'WHITE': 4,
         'MIDDLE EASTERN': 4,
         'PORTUGUESE': 4,
         'AMERICAN INDIAN': 0,
         'NATIVE HAWAIIAN': 0,
         'UNABLE TO OBTAIN': 0,
         'PATIENT DECLINED TO ANSWER': 0,
         'UNKNOWN': 0,
         'OTHER': 0,
         '': 0}


def transform_ethnicity(ethnicity_series):
    global e_map

    def aggregate_ethnicity(ethnicity_str):
        return ethnicity_str.replace(' OR ', '/').split(' - ')[0].split('/')[0]

    ethnicity_series = ethnicity_series.apply(aggregate_ethnicity)
    return {'Ethnicity': ethnicity_series.fillna('').apply(lambda s: e_map[s] if s in e_map else e_map['OTHER'])}

#################################################
# transform admission type
#################################################

admittype_map = {'EMERGENCY': 1,
         'ELECTIVE': 2,
         'NEWBORN': 3,
         'URGENT': 4,
         'OTHER': 0,
         '': 0}


def transform_admission_type(adim_series):
    global admittype_map
    return {'ADMISSION_TYPE': adim_series.fillna('').apply(lambda s: admittype_map[s] if s in admittype_map else admittype_map['OTHER'])}



#################################################
# filter
#################################################

# filter event time 24/48 hours after intime of icustay per patient
def filter_event_duration(events, duaration):
    events = events.copy()
    events = events[ ( (events['HOURS'] > 0) & (events['HOURS'] <= duaration) ) ]
    return events



#################################################
# Time series preprocessing
#################################################

def read_itemid_to_variable_map(fn, variable_column='LEVEL2'):
    var_map = dataframe_from_csv(fn, index_col=None).fillna('').astype(str)
    # var_map[variable_column] = var_map[variable_column].apply(lambda s: s.lower())
    var_map.COUNT = var_map.COUNT.astype(int)
    var_map = var_map[(var_map[variable_column] != '') & (var_map.COUNT > 0)]
    var_map = var_map[(var_map.STATUS == 'ready')]
    var_map.ITEMID = var_map.ITEMID.astype(int)
    var_map = var_map[[variable_column, 'ITEMID', 'MIMIC LABEL']].set_index('ITEMID')
    return var_map.rename({variable_column: 'VARIABLE', 'MIMIC LABEL': 'MIMIC_LABEL'}, axis=1)


def map_itemids_to_variables(events, var_map):
    return events.merge(var_map, left_on='ITEMID', right_index=True)


def read_variable_ranges(fn, variable_column='LEVEL2'):
    columns = [variable_column, 'OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER HIGH']
    to_rename = dict(zip(columns, [c.replace(' ', '_') for c in columns]))
    to_rename[variable_column] = 'VARIABLE'
    var_ranges = dataframe_from_csv(fn, index_col=None)
    # var_ranges = var_ranges[variable_column].apply(lambda s: s.lower())
    var_ranges = var_ranges[columns]
    var_ranges.rename(to_rename, axis=1, inplace=True)
    var_ranges = var_ranges.drop_duplicates(subset='VARIABLE', keep='first')
    var_ranges.set_index('VARIABLE', inplace=True)
    return var_ranges.loc[var_ranges.notnull().all(axis=1)]


# make multi-hot for icd of diagnosis
def extract_diagnosis_labels(diagnoses):
    global diagnosis_labels
    diagnoses['VALUE'] = 1
    labels = diagnoses[['ICUSTAY_ID', 'ICD9_CODE', 'VALUE']].drop_duplicates() \
        .pivot(index='ICUSTAY_ID', columns='ICD9_CODE', values='VALUE').fillna(0).astype(int)
    for l in diagnosis_labels:
        if l not in labels:
            labels[l] = 0
    labels = labels[diagnosis_labels]
    return labels.rename(dict(zip(diagnosis_labels, ['Diagnosis ' + d for d in diagnosis_labels])), axis=1)


def assemble_episodic_data(stays, diagnoses, procedures):
    data = {'Icustay': stays.ICUSTAY_ID, 'Age': stays.AGE, 'Length of Stay': stays.LengthofAdmit,
            'MORTALITY_INUNIT': stays.MORTALITY_INUNIT, 'MORTALITY': stays.MORTALITY,
            'MORTALITY_INHOSPITAL': stays.MORTALITY_INHOSPITAL,
            'MORTALITY_2DAYS': stays.MORTALITY_2DAYS, 'MORTALITY_3DAYS': stays.MORTALITY_3DAYS,
            'MORTALITY_30DAYS': stays.MORTALITY_30DAYS, 'MORTALITY_1YEAR': stays.MORTALITY_1YEAR,
            }
    data.update(transform_gender(stays.GENDER))
    data.update(transform_ethnicity(stays.ETHNICITY))
    data.update(transform_admission_type(stays.ADMISSION_TYPE))
    #data['Height'] = np.nan
    #data['Weight'] = np.nan
    data = pd.DataFrame(data).set_index('Icustay')
    data = data[['Ethnicity', 'Gender', 'Age', 'Length of Stay', 'ADMISSION_TYPE', \
                 'MORTALITY', 'MORTALITY_INUNIT', 'MORTALITY_INHOSPITAL', \
                 'MORTALITY_2DAYS', 'MORTALITY_3DAYS', \
                 'MORTALITY_30DAYS', 'MORTALITY_1YEAR',
                 ]]
    if diagnoses is not None:
        data = data.merge(diagnoses, left_index=True, right_index=True)
    if procedures is not None:
        data = data.merge(procedures, left_index=True, right_index=True)
    return data


def remove_outliers_for_variable(events, variable, ranges):
    if variable not in ranges.index:
        return events
    idx = (events.VARIABLE == variable)
    v = events.VALUE[idx].copy()
    v.loc[v < ranges.OUTLIER_LOW[variable]] = np.nan
    v.loc[v > ranges.OUTLIER_HIGH[variable]] = np.nan
    v.loc[v < ranges.VALID_LOW[variable]] = ranges.VALID_LOW[variable]
    v.loc[v > ranges.VALID_HIGH[variable]] = ranges.VALID_HIGH[variable]
    events.loc[idx, 'VALUE'] = v
    return events

################################################
######            clean event        ###########
################################################

# SBP: some are strings of type SBP/DBP
def clean_sbp(df):
    v = df.VALUE.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(1))
    return v.astype(float)


def clean_dbp(df):
    v = df.VALUE.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(2))
    return v.astype(float)


# CRR: strings with brisk, <3 normal, delayed, or >3 abnormal
def clean_crr(df):
    v = Series(np.zeros(df.shape[0]), index=df.index)
    v[:] = np.nan

    # when df.VALUE is empty, dtype can be float and comparision with string
    # raises an exception, to fix this we change dtype to str
    df_value_str = df.VALUE.astype(str)

    v.loc[(df_value_str == 'Normal <3 secs') | (df_value_str == 'Brisk')] = 0
    v.loc[(df_value_str == 'Abnormal >3 secs') | (df_value_str == 'Delayed')] = 1
    return v


# FIO2: many 0s, some 0<x<0.2 or 1<x<20
def clean_fio2(df):
    v = df.VALUE.astype(float).copy()

    ''' The line below is the correct way of doing the cleaning, since we will not compare 'str' to 'float'.
    If we use that line it will create mismatches from the data of the paper in ~50 ICU stays.
    The next releases of the benchmark should use this line.
    '''
    # idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (v>1.0)

    ''' The line below was used to create the benchmark dataset that the paper used. Note this line will not work
    in python 3, since it may try to compare 'str' to 'float'.
    '''
    # idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (df.VALUE > 1.0)

    ''' The two following lines implement the code that was used to create the benchmark dataset that the paper used.
    This works with both python 2 and python 3.
    '''
    is_str = np.array(map(lambda x: type(x) == str, list(df.VALUE)), dtype=np.bool)
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (is_str | (~is_str & (v > 1.0)))

    v.loc[idx] = v[idx] / 100.
    return v


# GLUCOSE, PH: sometimes have ERROR as value
def clean_lab(df):
    v = df.VALUE.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.loc[idx] = np.nan
    return v.astype(float)


# O2SAT: small number of 0<x<=1 that should be mapped to 0-100 scale
def clean_o2sat(df):
    # change "ERROR" to NaN
    v = df.VALUE.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.loc[idx] = np.nan

    v = v.astype(float)
    idx = (v <= 1)
    v.loc[idx] = v[idx] * 100.
    return v


# Temperature: map Farenheit to Celsius, some ambiguous 50<x<80
def clean_temperature(df):
    v = df.VALUE.astype(float).copy()
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'F' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'F' in s.lower()) | (v >= 79)
    v.loc[idx] = (v[idx] - 32) * 5. / 9
    return v


# Weight: some really light/heavy adults: <50 lb, >450 lb, ambiguous oz/lb
# Children are tough for height, weight
def clean_weight(df):
    v = df.VALUE.astype(float).copy()
    # ounces
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'oz' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'oz' in s.lower())
    v.loc[idx] = v[idx] / 16.
    # pounds
    idx = idx | df.VALUEUOM.fillna('').apply(lambda s: 'lb' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'lb' in s.lower())
    v.loc[idx] = v[idx] * 0.453592
    return v


# Height: some really short/tall adults: <2 ft, >7 ft)
# Children are tough for height, weight
def clean_height(df):
    v = df.VALUE.astype(float).copy()
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'in' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'in' in s.lower())
    v.loc[idx] = np.round(v[idx] * 2.54)
    return v


# ETCO2: haven't found yet
# Urine output: ambiguous units (raw ccs, ccs/kg/hr, 24-hr, etc.)
# Tidal volume: tried to substitute for ETCO2 but units are ambiguous
# Glascow coma scale eye opening
# Glascow coma scale motor response
# Glascow coma scale total
# Glascow coma scale verbal response
# Heart Rate
# Respiratory rate
# Mean blood pressure
clean_fns = {
    'Capillary refill rate': clean_crr,
    'Diastolic blood pressure': clean_dbp,
    'Systolic blood pressure': clean_sbp,
    'Fraction inspired oxygen': clean_fio2,
    'Oxygen saturation': clean_o2sat,
    'Glucose': clean_lab,
    'pH': clean_lab,
    'Temperature': clean_temperature,
    'Weight': clean_weight,
    'Height': clean_height
}


def clean_events(events):
    global clean_fns
    for var_name, clean_fn in clean_fns.items():
        idx = (events.VARIABLE == var_name)
        try:
            events.loc[idx, 'VALUE'] = clean_fn(events[idx])
        except Exception as e:
            import traceback
            print("Exception in clean_events:", clean_fn.__name__, e)
            print(traceback.format_exc())
            print("number of rows:", np.sum(idx))
            print("values:", events[idx])
            exit()
    return events.loc[events.VALUE.notnull()]




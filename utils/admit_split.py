import os
import csv
import datetime
import numpy as np
import pandas as pd


def read_admissions_table_all(mimic3_path):
    admits = dataframe_from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv'))
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits

def read_events_table_by_row(mimic3_path, table):
    reader = csv.DictReader(open(os.path.join(mimic3_path, table.upper() + '.csv'), 'r'))
    for i, row in enumerate(reader):
        if 'ICUSTAY_ID' not in row:
            row['ICUSTAY_ID'] = ''
        yield row, i


def read_events_table_and_break_up_by_admit(mimic3_path, table, output_path,
                                            admits_to_keep=None):
    nb_rows_dict = {'inputevents_cv':1483111,'inputevents_mv':1732207,'chartevents': 33866825, 'labevents': 19645103, 'outputevents': 180922}

    nb_rows = nb_rows_dict[table.lower()]
    obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE']


    if admits_to_keep is not None:
        admits_to_keep = set([str(s) for s in admits_to_keep])

    if admits_to_keep is None:
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

    for row, row_no in tqdm(read_events_table_by_row(mimic3_path, table), desc='Processing {} table'.format(table)):

        if (int(row['HADM_ID']) not in admits_to_keep):
            continue
        ###################################################
        row_out = {'SUBJECT_ID': row['SUBJECT_ID'],
                'HADM_ID': row['HADM_ID'],
                'ICUSTAY_ID': row['ICUSTAY_ID'],
                'CHARTTIME': row['CHARTTIME'],
                'ITEMID': row['ITEMID'],
                'VALUE': row['VALUENUM']}
        ###################################################
        if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['HADM_ID']:
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['HADM_ID']

    if data_stats.curr_subject_id != '':
        write_current_observations()





def read_events_table_and_generate_matrix_of_time_series(res_path, table, output_path,
                                            admits_to_keep=None):
    if admits_to_keep is not None:
        admits_to_keep = set([str(s) for s in admits_to_keep])

    if admits_to_keep is None:
        return None

    valid_all = []
    valid_chart_tomean = []
    valid_lab_tomean = []

    v = np.load(os.path.join(res_path, 'filtered_input.npy'), allow_pickle=True).tolist()
    valid_input = v['id']
    valid_all.extend(valid_input)

    v = np.load(os.path.join(res_path, 'filtered_output.npy'), allow_pickle=True).tolist()
    valid_output = v['id']
    valid_all.extend(valid_output)

    v = np.load(os.path.join(res_path, 'filtered_chart.npy'), allow_pickle=True).tolist()
    valid_chart = v['id']
    valid_chart_tomean.extend(valid_chart)
    valid_all.extend(valid_chart)

    v = np.load(os.path.join(res_path, 'filtered_chart_num.npy'), allow_pickle=True).tolist()
    valid_chart_num = v['id']
    valid_chart_tomean.extend(valid_chart_num)
    valid_all.extend(valid_chart_num)

    v = np.load(os.path.join(res_path, 'filtered_chart_cate.npy'), allow_pickle=True).tolist()
    valid_chart_cate = v['id']
    valid_all.extend(valid_chart_cate)

    v = np.load(os.path.join(res_path, 'filtered_chart_ratio.npy'), allow_pickle=True).tolist()
    valid_chart_ratio = v['id']
    valid_chart_tomean.extend(valid_chart_ratio)
    valid_all.extend(valid_chart_ratio)

    v = np.load(os.path.join(res_path, 'filtered_lab.npy'), allow_pickle=True).tolist()
    valid_lab = v['id']
    valid_lab_tomean.extend(valid_lab)
    valid_all.extend(valid_lab)

    v = np.load(os.path.join(res_path, 'filtered_lab_num.npy'), allow_pickle=True).tolist()
    valid_lab_num = v['id']
    valid_lab_tomean.extend(valid_lab_num)
    valid_all.extend(valid_lab_num)

    v = np.load(os.path.join(res_path, 'filtered_lab_cate.npy'), allow_pickle=True).tolist()
    valid_lab_cate = v['id']
    valid_all.extend(valid_lab_cate)

    v = np.load(os.path.join(res_path, 'filtered_lab_ratio.npy'), allow_pickle=True).tolist()
    valid_lab_ratio = v['id']
    valid_lab_tomean.extend(valid_lab_ratio)
    valid_all.extend(valid_lab_ratio)

    v = np.load(os.path.join(res_path, 'filtered_prescript.npy'), allow_pickle=True).tolist()
    valid_prescript = v['id']
    valid_all.extend(valid_prescript)

    ls = os.listdir(output_path)
    if table.lower() in ["inputevents", "outputevents", "prescriptions"]:
        for hid in tqdm(ls, total=len(ls)):
            fn = os.path.join(output_path, hid, table.lower() + '.csv')
            events = pd.read_csv(fn, low_memory=False)
            metadata = events[['CHARTTIME', 'ICUSTAY_ID']].sort_values(by=['CHARTTIME', 'ICUSTAY_ID']) \
                .drop_duplicates(keep='first').set_index('CHARTTIME')
            timeseries = events[['CHARTTIME', "ITEMID", 'VALUE']] \
                .fillna(0) \
                .sort_values(by=['CHARTTIME', "ITEMID", 'VALUE'], axis=0)

            duplcate_row = timeseries.duplicated(subset=['CHARTTIME', "ITEMID"], keep=False)
            duplicate_data = timeseries.loc[duplcate_row, :]
            duplicate_data_sum = duplicate_data.groupby(by=['CHARTTIME', "ITEMID"]).agg(
                [("VALUE", 'sum')]).reset_index()
            duplicate_data_sum.columns = duplicate_data_sum.columns.droplevel(1)
            no_duplicate = timeseries.drop_duplicates(subset=['CHARTTIME', "ITEMID"], keep=False)

            timeseries_processed = pd.concat([no_duplicate, duplicate_data_sum], sort=False)
            timeseries_processed = timeseries_processed.sort_values(by=['CHARTTIME', "ITEMID"]) \
                .pivot(index='CHARTTIME', columns="ITEMID", values='VALUE') \
                .merge(metadata, left_index=True, right_index=True) \
                .sort_index(axis=0).reset_index()

            for v in valid_all:
                if v not in timeseries_processed:
                    timeseries_processed[v] = np.nan
            timeseries_processed.columns = timeseries_processed.columns.astype(str)
            columns = list(timeseries_processed.columns)
            columns_sorted = sorted(columns, key=(lambda x: "" if x == "HOURS" else x))
            timeseries_processed = timeseries_processed[columns_sorted]

            timeseries_processed.to_csv(fn.replace(".csv", "_ts.csv"))

    if table.lower() in ["labevents", "chartevents"]:
        if table.lower in ["chartevents"]:
            valid_tomean = valid_chart_tomean
        else:
            valid_tomean = valid_lab_tomean
        for hid in tqdm(ls, total=len(ls)):
            fn = os.path.join(output_path, hid, table.lower() + '.csv')
            events = pd.read_csv(fn, low_memory=False)
            metadata = events[['CHARTTIME', 'ICUSTAY_ID']].sort_values(by=['CHARTTIME', 'ICUSTAY_ID']) \
                .drop_duplicates(keep='first').set_index('CHARTTIME')
            ###########################################################
            timeseries_num = events.query(f"ITEMID in {valid_tomean}")
            timeseries_num = timeseries_num[['CHARTTIME', "ITEMID", 'VALUE']] \
                .fillna(0) \
                .sort_values(by=['CHARTTIME', "ITEMID", 'VALUE'], axis=0)

            duplcate_row = timeseries_num.duplicated(subset=['CHARTTIME', "ITEMID"], keep=False)
            duplicate_data = timeseries_num.loc[duplcate_row, :]
            duplicate_data_sum = duplicate_data.groupby(by=['CHARTTIME', "ITEMID"]).agg(
                [("VALUE", 'mean')]).reset_index()
            duplicate_data_sum.columns = duplicate_data_sum.columns.droplevel(1)
            no_duplicate = timeseries_num.drop_duplicates(subset=['CHARTTIME', "ITEMID"], keep=False)
            timeseries_num = pd.concat([no_duplicate, duplicate_data_sum], sort=False)
            ###########################################################
            timeseries_cat = events.query(f"ITEMID in {valid_chart_cate}")
            timeseries_cat = timeseries_cat[["CHARTTIME", "ITEMID", 'VALUE']] \
                .drop_duplicates(["CHARTTIME", "ITEMID"], keep='first')
            ###########################################################
            timeseries_processed = pd.concat([timeseries_num, timeseries_cat], sort=False).sort_values(by=['CHARTTIME', "ITEMID"]) \
                .pivot(index='CHARTTIME', columns="ITEMID", values='VALUE') \
                .merge(metadata, left_index=True, right_index=True) \
                .sort_index(axis=0).reset_index()

            for v in valid_all:
                if v not in timeseries_processed:
                    timeseries_processed[v] = np.nan
            timeseries_processed.columns = timeseries_processed.columns.astype(str)
            columns = list(timeseries_processed.columns)
            columns_sorted = sorted(columns, key=(lambda x: "" if x == "HOURS" else x))
            timeseries_processed = timeseries_processed[columns_sorted]

            timeseries_processed.to_csv(fn.replace(".csv", "_ts.csv"))

def read_admits_and_add_age_los_mortality_icd9(mimic3_path, output_admit_path, admission_ids):
    cate = ['admission_type', 'admission_location', 'insurance', 'language', 'religion', 'marital_status', 'ethnicity']
    mapping = np.load(os.path.join(res_path, 'adm_catemappings.npy'), allow_pickle=True).tolist()
    admits = pd.read_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv'), low_memory=False)
    patients = pd.read_csv(os.path.join(mimic3_path, 'PATIENTS.csv'), low_memory=False)
    stays = pd.read_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv'), low_memory=False)
    services = pd.read_csv(os.path.join(mimic3_path, 'SERVICES.csv'), low_memory=False)
    diagnoses = pd.read_csv(os.path.join(mimic3_path, 'DIAGNOSES_ICD.csv'), low_memory=False)
    for aid in tqdm(admission_ids, total=len(admission_ids)):
        row = admits[admits.HADM_ID == int(aid)].iloc[0]
        subject_id = row["SUBJECT_ID"]
        admittime = row["ADMITTIME"]
        dischtime = row["DISCHTIME"]
        deathtime = row["DEATHTIME"]
        admittime = datetime.datetime.strptime(admittime, "%Y-%m-%d %H:%M:%S")
        dischtime = datetime.datetime.strptime(dischtime, "%Y-%m-%d %H:%M:%S")
        deathtime = datetime.datetime.strptime(deathtime, "%Y-%m-%d %H:%M:%S")

        stay = stays[stays.HADM_ID == int(aid)].sort_values(["INTIME"]).iloc[0]
        intime = stay["INTIME"]
        if intime is not None:
            admittime = datetime.datetime.strptime(intime, "%Y-%m-%d %H:%M:%S")

        patient = patients[patients.SUBJECT_ID == int(subject_id)].iloc[0]
        birthdate = patient["DOB"]
        final_deathdate = patient["DOD"]
        birthdate = datetime.datetime.strptime(birthdate, "%Y-%m-%d %H:%M:%S")
        final_deathdate = datetime.datetime.strptime(final_deathdate, "%Y-%m-%d %H:%M:%S")


        service = services[services.HADM_ID == int(aid)].iloc[0]
        curr_service = service["CURR_SERVICE"]

        mortal = 0
        labelGuarantee = 0
        die24 = 0
        die24_48 = 0
        die48_72 = 0
        die30days = 0
        die1year = 0
        if (deathtime != None):
            mortal = 1
            if (deathtime != dischtime):
                labelGuarantee = 1
            secnum = (deathtime - admittime).total_seconds()
            if secnum <= 24 * 60 * 60:
                die24 = 1
            if secnum <= 48 * 60 * 60:
                die24_48 = 1
            if secnum <= 72 * 60 * 60:
                die48_72 = 1
        if dischtime is not None and final_deathdate is not None:
            dischsecnum = (final_deathdate - dischtime).total_seconds()
            if dischsecnum <= 30 * 24 * 60 * 60:
                die30days = 1
            if dischsecnum <= 365 * 24 * 60 * 60:
                die1year = 1

        if curr_service is None:
            curr_service = 'NB'

        data = [aid, subject_id, (admittime - birthdate).total_seconds() / (3600 * 24),
                (dischtime - admittime).total_seconds() // 60., mortal, labelGuarantee, die24, die24_48, die48_72,
                die30days, die1year, mapping['curr_service'][curr_service]]
        for i in range(5, 12):
            data.append(mapping[cate[i - 5]][admission[i]])

        diagnose = diagnoses.query(f"HADM_ID == {int(aid)}")

        list_icd9 = []
        for i, r in diagnose.iterrows():
            icd = r["ICD9_CODE"]
            if icd is None:
                continue
            if (icd[0] == 'V'):
                label_name = 19
                numstr = icd[0:3] + '.' + icd[3:len(icd)]
            elif (icd[0] == 'E'):
                cate20 += 1
                label_name = 20
                numstr = icd
            else:
                num = float(icd[0:3])
                numstr = icd[0:3] + '.' + icd[3:len(icd)]
                if (num >= 1 and num <= 139):
                    label_name = 0
                if (num >= 140 and num <= 239):
                    label_name = 1
                if (num >= 240 and num <= 279):
                    label_name = 2
                if (num >= 280 and num <= 289):
                    label_name = 3
                if (num >= 290 and num <= 319):
                    label_name = 4
                if (num >= 320 and num <= 389):
                    label_name = 5
                if (num >= 390 and num <= 459):
                    label_name = 6
                if (num >= 460 and num <= 519):
                    label_name = 7
                if (num >= 520 and num <= 579):
                    label_name = 8
                if (num >= 580 and num <= 629):
                    label_name = 9
                if (num >= 630 and num <= 677):
                    label_name = 10
                if (num >= 680 and num <= 709):
                    label_name = 11
                if (num >= 710 and num <= 739):
                    label_name = 12
                if (num >= 740 and num <= 759):
                    label_name = 13
                if (num >= 760 and num <= 779):
                    label_name = 14
                if (num >= 780 and num <= 789):
                    label_name = 15
                if (num >= 790 and num <= 796):
                    label_name = 16
                if (num >= 797 and num <= 799):
                    label_name = 17
                if (num >= 800 and num <= 999):
                    label_name = 18
            list_icd9.append([aid, icd, numstr, label_name])


        fn = os.path.join(output_admit_path, str(aid)+'_static.npy')
        fn2 = os.path.join(output_admit_path, str(aid)+'_icd9.npy')
        np.save(fn, data)
        np.save(fn2, list_icd9)


def process_patient(aid, output_admit_path):
    with open(os.path.join(output_admit_path,'adm-{0}.log').format(str('%.6d' % aid)), 'w') as f:
        try:
            proc = processing(aid, f)
            if len(proc) == 0:
                return
            res = {
                'timeseries': sparsify(proc),
                'general': ageLosMortality(aid, f),
                'icd9': ICD9(aid, f)
            }
            np.save(os.path.join(output_admit_path,'adm-' + str('%.6d' % aid)), res)
            print('finished {0}!'.format(aid))
        except Exception as e:
            print('failed at {0}!'.format(aid))



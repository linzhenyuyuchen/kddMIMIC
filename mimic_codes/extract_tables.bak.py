import os, re, sys, yaml, pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from utils.subject_split import *


def extract_itemids_events(mimic3_path, output_path, variable_map_path, table):
    print(table)
    nb_rows_dict = {'inputevents_cv':17527936,'inputevents_mv':3618992, 'outputevents': 4349219, 'labevents': 27854056, 'chartevents': 330712484}
    nb_rows = nb_rows_dict[table.lower()]
    obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'VALUEUOM']
    #################################################################
    df = pd.read_csv(os.path.join(variable_map_path, "130features.csv"))
    if table.lower() in ["inputevents_cv", "inputevents_mv"]:
        itemids_df = df.query("table=='input'").item_id
    elif table.lower() in ["labevents"]:
        itemids_df = df.query("table=='lab'").item_id
    elif table.lower() in ["outputevents"]:
        itemids_df = df.query("table=='output'").item_id
    elif table.lower() in ["chartevents"]:
        itemids_df = df.query("table=='chart'").item_id
    itemids = []
    for i in itemids_df:
        itemids = itemids + i.split(",")
    itemids = list(set(itemids))
    if "" in itemids:
        itemids.remove("")
    #print(itemids)
    itemids = [int(x.strip()) for x in itemids]
    #################################################################
    with open("./resources/itemid_uom_map/unit_coeffient.pkl", "rb") as f:
        unit_co = pickle.load(f)
    #################################################################
    time_name = "CHARTTIME"
    value_name = "VALUENUM"
    valueuom_name = "VALUEUOM"
    if table.lower() in ["inputevents_cv"]:
        time_name = "CHARTTIME"
        value_name = "AMOUNT"
        valueuom_name = "AMOUNTUOM"
    elif table.lower() in ["inputevents_mv"]:
        time_name = "ENDTIME"
        value_name = "AMOUNT"
        valueuom_name = "AMOUNTUOM"
    elif table.lower() in ["outputevents"]:
        time_name = "CHARTTIME"
        value_name = "VALUE"
        valueuom_name = "VALUEUOM"
    elif table.lower() in ["chartevents", "labevents"]:
        time_name = "CHARTTIME"
        value_name = "VALUENUM"
        valueuom_name = "VALUEUOM"
    #################################################################
    fn = os.path.join(output_path, table.upper()+".csv")
    if os.path.exists(fn) or os.path.isfile(fn):
        os.remove(fn)
    else:
        f = open(fn, 'w')
        f.write(','.join(obs_header) + '\n')
        f.close()
    w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
    #################################################################
    pbar = tqdm(read_events_table_by_row(mimic3_path, table.lower()),total=nb_rows)
    c = 0
    count_unit_transform = 0
    for row, row_no in pbar:
        if not int(row["ITEMID"]) in itemids:
            continue
        ###############################################
        if pd.isnull(row[value_name]) or row[value_name] == '':
            continue
        ###############################################
        amount = float(row[value_name])
        strs = f'{row["ITEMID"]}-{row[valueuom_name].lower()}'
        if strs in unit_co:
            count_unit_transform += 1
            num = unit_co[strs]
            if num != 0:
                amount = amount / num
            else:
                continue
        ###############################################
        c += 1
        pbar.set_description("C-%s"%c)
        data_stats = [{'SUBJECT_ID': row["SUBJECT_ID"], 'HADM_ID': row["HADM_ID"], 'ICUSTAY_ID': row["ICUSTAY_ID"], 'ITEMID': row["ITEMID"], \
                       'CHARTTIME': row[time_name], 'VALUENUM': amount, 'VALUEUOM': row[valueuom_name]}]
        w.writerows(data_stats)
    print(f"There are {c} rows of records added!")
    print(f"There are {count_unit_transform} units changed!")

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

def extract_itemids_prescriptions(mimic3_path, output_path, variable_map_path):
    table = "prescriptions"
    print(table)
    nb_rows = 4156450
    obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'VALUEUOM']
    #################################################################
    itemids_df = pd.read_csv(os.path.join(variable_map_path, "130features.csv"))
    itemids_df = itemids_df.query("table=='pres'").item_id
    itemids = []
    for i in itemids_df:
        itemids = itemids + i.split(",")
    itemids = list(set(itemids))
    if "" in itemids:
        itemids.remove("")
    itemids = [x.strip() for x in itemids]
    #################################################################
    fn = os.path.join(output_path, table.upper()+".csv")
    if os.path.exists(fn) or os.path.isfile(fn):
        os.remove(fn)
    else:
        f = open(fn, 'w')
        f.write(','.join(obs_header) + '\n')
        f.close()
    w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
    #################################################################
    pbar = tqdm(read_events_table_by_row(mimic3_path, table.lower()),total=nb_rows)
    c = 0
    for row, row_no in pbar:
        drug_tmp = str(row["FORMULARY_DRUG_CD"])
        ###############################################
        if drug_tmp not in itemids:
            continue
        if pd.isnull(row["DOSE_VAL_RX"]) or row["DOSE_VAL_RX"] == '':
            continue
        if drug_tmp == "INHRIV" and row["DOSE_UNIT_RX"] == "UNIT/HR":
            continue
        if drug_tmp == "BISA5" and row["DOSE_UNIT_RX"] == "ml":
            continue
        if drug_tmp == "INSULIN" and row["DOSE_UNIT_RX"] != "UNIT":
            continue
        ###############################################
        amount = filter_value(str(row["DOSE_VAL_RX"]))
        ###############################################
        c += 1
        pbar.set_description("C-%s"%c)
        data_stats = [{'SUBJECT_ID': row["SUBJECT_ID"], 'HADM_ID': row["HADM_ID"], 'ICUSTAY_ID': row["ICUSTAY_ID"], 'ITEMID': drug_tmp, \
                       'CHARTTIME': row["STARTDATE"], 'VALUENUM': amount, 'VALUEUOM': row["DOSE_UNIT_RX"]}]
        w.writerows(data_stats)


def main():
    cfg = yaml.load(open("./config.yaml","r"), Loader=yaml.FullLoader)
    mimic3_path = cfg["mimic3_path"]
    output_path = cfg["output_path"]
    variable_map_path = cfg["variable_map"]
    tables = cfg["event_tables"]
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    extract_itemids_prescriptions(mimic3_path, output_path, variable_map_path)
    for table in tables:
        if table in  [ "inputevents_cv", "inputevents_mv", "outputevents", "labevents", "chartevents"]:
            extract_itemids_events(mimic3_path, output_path, variable_map_path, table)

if __name__ == "__main__":
    main()

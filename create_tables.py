import os, re, sys, yaml, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from utils.subject_split import *
from utils.extract_tables_custom import *


def main():
    cfg = yaml.load(open("./config.yaml","r"), Loader=yaml.FullLoader)
    mimic3_path = cfg["mimic3_path"]
    output_path = cfg["output_path"]
    res_path = cfg["res_path"]
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    UNITSMAP = parseUnitsMap(os.path.join(res_path, 'unitsmap.unit'))
    ##########################################################
    _adm = np.load(os.path.join(res_path, 'admission_ids.npy'), allow_pickle=True).tolist()
    admission_ids = _adm['admission_ids']
    admission_ids_txt = _adm['admission_ids_txt']

    _adm_first = np.load(os.path.join(res_path, 'admission_first_ids.npy'), allow_pickle=True).tolist()
    admission_first_ids = _adm['admission_ids']

    catedict = np.load(os.path.join(res_path, 'catedict.npy'), allow_pickle=True).tolist()

    ##########################################################

    v = np.load(os.path.join(res_path, 'filtered_input.npy'), allow_pickle=True).tolist()
    valid_input = v['id']
    valid_input_unit = v['unit']

    v = np.load(os.path.join(res_path, 'filtered_output.npy'), allow_pickle=True).tolist()
    valid_output = v['id']

    v = np.load(os.path.join(res_path, 'filtered_chart.npy'), allow_pickle=True).tolist()
    valid_chart = v['id']
    valid_chart_unit = v['unit']

    v = np.load(os.path.join(res_path, 'filtered_chart_num.npy'), allow_pickle=True).tolist()
    valid_chart_num = v['id']
    valid_chart_num_unit = v['unit']

    v = np.load(os.path.join(res_path, 'filtered_chart_cate.npy'), allow_pickle=True).tolist()
    valid_chart_cate = v['id']

    v = np.load(os.path.join(res_path, 'filtered_chart_ratio.npy'), allow_pickle=True).tolist()
    valid_chart_ratio = v['id']

    v = np.load(os.path.join(res_path, 'filtered_lab.npy'), allow_pickle=True).tolist()
    valid_lab = v['id']
    valid_lab_unit = v['unit']

    v = np.load(os.path.join(res_path, 'filtered_lab_num.npy'), allow_pickle=True).tolist()
    valid_lab_num = v['id']
    valid_lab_num_unit = v['unit']

    v = np.load(os.path.join(res_path, 'filtered_lab_cate.npy'), allow_pickle=True).tolist()
    valid_lab_cate = v['id']

    v = np.load(os.path.join(res_path, 'filtered_lab_ratio.npy'), allow_pickle=True).tolist()
    valid_lab_ratio = v['id']

    v = np.load(os.path.join(res_path, 'filtered_microbio.npy'), allow_pickle=True).tolist()
    valid_microbio = v['id']

    v = np.load(os.path.join(res_path, 'filtered_prescript.npy'), allow_pickle=True).tolist()
    valid_prescript = v['id']
    valid_prescript_unit = v['unit']

    allids = valid_input + valid_output + valid_chart + valid_chart_num + valid_chart_cate + valid_chart_ratio + valid_chart_ratio + valid_lab + valid_lab_num + valid_lab_cate + valid_lab_ratio + valid_lab_ratio + valid_microbio + valid_prescript
    print("# of all available itemids", len(set(allids)))
    ##########################################################
    # map itemids to [0..n] column
    index = 0
    map_itemid_index = {}
    allitem = allids
    allitem_unit = valid_input_unit + ['NOCHECK'] * len(valid_output) + valid_chart_unit + valid_chart_num_unit + [
        'NOCHECK'] * len(valid_chart_cate) + ['NOCHECK'] * 2 * len(
        valid_chart_ratio) + valid_lab_unit + valid_lab_num_unit + ['NOCHECK'] * len(valid_lab_cate) + [
                       'NOCHECK'] * 2 * len(valid_lab_ratio) + ['NOCHECK'] * len(valid_microbio) + valid_prescript_unit
    for i in range(len(allitem_unit)):
        allitem_unit[i] = allitem_unit[i].replace(' ', '').lower()
    assert len(allitem) == len(allitem_unit)
    for ai in allitem:
        if ai not in map_itemid_index.keys():
            map_itemid_index[ai] = [index]
        else:
            map_itemid_index[ai].append(index)
        index += 1
    ##########################################################
    nb_rows_dict = {'inputevents_cv':17527936,'inputevents_mv':3618992, 'outputevents': 4349219, 'labevents': 27854056, 'chartevents': 330712484}
    obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'VALUEUOM']
    ##########################################################
    ################      inputevents_mv      ################
    ##########################################################
    """
    table = "inputevents_mv"
    nb_rows = nb_rows_dict[table.lower()]
    print("creating ", table)
    fn = os.path.join(output_path, table.upper()+".csv")
    if os.path.exists(fn) or os.path.isfile(fn):
        os.remove(fn)
    else:
        f = open(fn, 'w')
        f.write(','.join(obs_header) + '\n')
        f.close()
    w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
    #################################################################
    c = 0
    unitmap = UNITSMAP['inputevents']
    pbar = tqdm(read_events_table_by_row(mimic3_path, table.lower()),total=nb_rows)
    for row, row_no in pbar:
        if pd.isnull(row["ENDTIME"]) or pd.isnull(row["AMOUNT"]):
            continue
        ###############################################
        starttime, itemid, amount, amountuom = row["ENDTIME"], row["ITEMID"], row["AMOUNT"], row["AMOUNTUOM"]
        itemid = int(itemid)
        if not itemid in valid_input:
            continue
        amountuom = amountuom.replace(' ', '').lower()
        mainunit = allitem_unit[map_itemid_index[itemid][0]]
        ###############################################
        if itemid in unitmap.keys():
            dst_value = convert_units(unitmap[itemid], amountuom, mainunit, amount)
        else:
            if amountuom == mainunit:
                dst_value = float(amount)
            else:
                dst_value = None
        if dst_value is None:
            print('not convertible: ', row_no)
            continue
        ###############################################
        c += 1
        pbar.set_description("C-%s"%c)
        data_stats = [{'SUBJECT_ID': row["SUBJECT_ID"], 'HADM_ID': row["HADM_ID"], 'ICUSTAY_ID': row["ICUSTAY_ID"], 'ITEMID': itemid, \
                       'CHARTTIME': starttime, 'VALUENUM': dst_value, 'VALUEUOM': mainunit}]
        w.writerows(data_stats)
    """
    ##########################################################
    ################      inputevents_cv      ################
    ##########################################################
    """
    table = "inputevents_cv"
    nb_rows = nb_rows_dict[table.lower()]
    print("creating ", table)
    fn = os.path.join(output_path, table.upper()+".csv")
    if os.path.exists(fn) or os.path.isfile(fn):
        os.remove(fn)
    else:
        f = open(fn, 'w')
        f.write(','.join(obs_header) + '\n')
        f.close()
    w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
    #################################################################
    c = 0
    unitmap = UNITSMAP['inputevents']
    pbar = tqdm(read_events_table_by_row(mimic3_path, table.lower()),total=nb_rows)
    for row, row_no in pbar:
        if pd.isnull(row["CHARTTIME"]) or pd.isnull(row["AMOUNT"]) or row["AMOUNT"] == '':
            continue
        ###############################################
        starttime, itemid, amount, amountuom = row["CHARTTIME"], row["ITEMID"], row["AMOUNT"], row["AMOUNTUOM"]
        itemid = int(itemid)
        if not itemid in valid_input:
            continue
        amountuom = amountuom.replace(' ', '').lower()
        mainunit = allitem_unit[map_itemid_index[itemid][0]]
        ###############################################
        if itemid in unitmap.keys():
            dst_value = convert_units(unitmap[itemid], amountuom, mainunit, amount)
        else:
            if amountuom == mainunit:
                dst_value = float(amount)
            else:
                dst_value = None
        if dst_value is None:
            print('not convertible: ', row_no)
            continue
        ###############################################
        c += 1
        pbar.set_description("C-%s"%c)
        data_stats = [{'SUBJECT_ID': row["SUBJECT_ID"], 'HADM_ID': row["HADM_ID"], 'ICUSTAY_ID': row["ICUSTAY_ID"], 'ITEMID': itemid, \
                       'CHARTTIME': starttime, 'VALUENUM': dst_value, 'VALUEUOM': mainunit}]
        w.writerows(data_stats)

    print("create ", table, " done!")
    ##########################################################
    ################       outputevents        ###############
    ##########################################################
    # We only need to discard records without starttime or value.
    table = "outputevents"
    nb_rows = nb_rows_dict[table.lower()]
    print("creating ", table)
    fn = os.path.join(output_path, table.upper()+".csv")
    if os.path.exists(fn) or os.path.isfile(fn):
        os.remove(fn)
    else:
        f = open(fn, 'w')
        f.write(','.join(obs_header) + '\n')
        f.close()
    w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
    #################################################################
    c = 0
    pbar = tqdm(read_events_table_by_row(mimic3_path, table.lower()),total=nb_rows)
    for row, row_no in pbar:
        if not int(row["ITEMID"]) in valid_output:
            continue
        if pd.isnull(row["CHARTTIME"]) or pd.isnull(row["VALUE"]):
            continue
        ###############################################
        c += 1
        pbar.set_description("C-%s"%c)
        data_stats = [{'SUBJECT_ID': row["SUBJECT_ID"], 'HADM_ID': row["HADM_ID"], 'ICUSTAY_ID': row["ICUSTAY_ID"], \
                       'ITEMID': row["ITEMID"], 'CHARTTIME': row["CHARTTIME"],\
                       'VALUENUM': row["VALUE"], 'VALUEUOM': row["VALUEUOM"]}]
        w.writerows(data_stats)

    print("create ", table, " done!")
    """
    ##########################################################
    ################      chartevents      ################
    ##########################################################
    table = "chartevents"
    nb_rows = nb_rows_dict[table.lower()]
    print("creating ", table)
    fn = os.path.join(output_path, table.upper()+".csv")
    if os.path.exists(fn) or os.path.isfile(fn):
        os.remove(fn)
    else:
        f = open(fn, 'w')
        f.write(','.join(obs_header) + '\n')
        f.close()
    w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
    #################################################################
    c = 0
    unitmap = UNITSMAP['chartevents']
    pbar = tqdm(read_events_table_by_row(mimic3_path, table.lower()),total=nb_rows)
    for row, row_no in pbar:
        starttime, itemid, value, valuenum, valueuom = row["CHARTTIME"], row["ITEMID"],  row["VALUE"], row["VALUENUM"], row["VALUEUOM"]
        itemid = int(itemid)
        if pd.isnull(starttime):
            continue
        ###############################################
        # case 1
        if itemid in valid_chart:
            if valuenum is None or valuenum=='':
                continue
            if valueuom is None:
                valueuom = ''
            valueuom = valueuom.replace(' ', '').lower()
            mainunit = allitem_unit[map_itemid_index[itemid][0]]
            ###############################################
            if itemid in unitmap.keys():
                dst_value = convert_units(unitmap[itemid], valueuom, mainunit, valuenum)
            else:
                if valueuom == mainunit or valueuom == '':
                    dst_value = float(valuenum)
                else:
                    dst_value = None
            if dst_value is None:
                print('not convertible: ', row_no)
                continue
            c += 1
            pbar.set_description("C-%s"%c)
            data_stats = [{'SUBJECT_ID': row["SUBJECT_ID"], 'HADM_ID': row["HADM_ID"], 'ICUSTAY_ID': row["ICUSTAY_ID"], 'ITEMID': itemid, \
                           'CHARTTIME': starttime, 'VALUENUM': dst_value, 'VALUEUOM': mainunit}]
            w.writerows(data_stats)
        ###############################################
        # case 2
        elif itemid in valid_chart_cate:
            if value is None or value == '':
                print('no value: ', row_no)
                continue
            value = catedict[itemid][value]
            if value is None:
                continue
            c += 1
            pbar.set_description("C-%s"%c)
            data_stats = [{'SUBJECT_ID': row["SUBJECT_ID"], 'HADM_ID': row["HADM_ID"], 'ICUSTAY_ID': row["ICUSTAY_ID"], 'ITEMID': itemid, \
                           'CHARTTIME': starttime, 'VALUENUM': value, 'VALUEUOM': valueuom}]
            w.writerows(data_stats)
        ###############################################
        # case 3
        elif itemid in valid_chart_num:
            if value is None or value == '':
                print('no value: ', row_no)
                continue
            ce2res = parseNum(value)
            if ce2res is None:
                print('not parsed')
                continue
            else:
                value = ce2res
            if valueuom is None:
                valueuom = ''
            currentunit = valueuom.replace(' ', '').replace('<', '').replace('>', '').replace('=', '').lower()
            mainunit = allitem_unit[map_itemid_index[itemid][0]]
            if currentunit == mainunit or currentunit == '':
                pass
            else:
                if itemid in unitmap.keys():
                    value = convert_units(unitmap[itemid], currentunit, mainunit, value)
                else:
                    if currentunit != mainunit:
                        value = None
            if value is None:
                print('not convertible: ', row_no)
                continue
            c += 1
            pbar.set_description("C-%s"%c)
            data_stats = [{'SUBJECT_ID': row["SUBJECT_ID"], 'HADM_ID': row["HADM_ID"], 'ICUSTAY_ID': row["ICUSTAY_ID"], 'ITEMID': itemid, \
                           'CHARTTIME': starttime, 'VALUENUM': value, 'VALUEUOM': valueuom}]
            w.writerows(data_stats)
        ###############################################
        # case 4
        elif itemid in valid_chart_ratio:
            if value is None or value=='':
                print('no value: ', row_no)
                continue
            try:
                fs = value.split('/')
                f1, f2 = fs[0], fs[1]
                if f1 != '':
                    value = float(f1)
                if f2 != '':
                    value = float(f2)
            except:
                print('not parsed: ', row_no)
                continue
            c += 1
            pbar.set_description("C-%s"%c)
            data_stats = [{'SUBJECT_ID': row["SUBJECT_ID"], 'HADM_ID': row["HADM_ID"], 'ICUSTAY_ID': row["ICUSTAY_ID"], 'ITEMID': itemid, \
                           'CHARTTIME': starttime, 'VALUENUM': value, 'VALUEUOM': valueuom}]
            w.writerows(data_stats)

    print("create ", table, " done!")
    ##########################################################
    ################      labevents      ################
    ##########################################################
    table = "labevents"
    nb_rows = nb_rows_dict[table.lower()]
    print("creating ", table)
    fn = os.path.join(output_path, table.upper()+".csv")
    if os.path.exists(fn) or os.path.isfile(fn):
        os.remove(fn)
    else:
        f = open(fn, 'w')
        f.write(','.join(obs_header) + '\n')
        f.close()
    w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
    #################################################################
    c = 0
    unitmap = UNITSMAP['labevents']
    pbar = tqdm(read_events_table_by_row(mimic3_path, table.lower()),total=nb_rows)
    for row, row_no in pbar:
        starttime, itemid, value, valuenum, valueuom = row["CHARTTIME"], row["ITEMID"],  row["VALUE"], row["VALUENUM"], row["VALUEUOM"]
        itemid = int(itemid)
        if pd.isnull(starttime):
            continue
        ###############################################
        # case 1
        if itemid in valid_lab:
            if valuenum is None or valuenum =='':
                continue
            if valueuom is None:
                valueuom = ''
            valueuom = valueuom.replace(' ','').replace('<','').replace('>','').replace('=','').lower()
            mainunit = allitem_unit[map_itemid_index[itemid][0]]
            ###############################################
            if itemid in unitmap.keys():
                dst_value = convert_units(unitmap[itemid], valueuom, mainunit, valuenum)
            else:
                if valueuom == mainunit or valueuom == '':
                    dst_value = float(valuenum)
                else:
                    dst_value = None
            if dst_value is None:
                print('not convertible: ', row_no)
                continue
            c += 1
            pbar.set_description("C-%s"%c)
            data_stats = [{'SUBJECT_ID': row["SUBJECT_ID"], 'HADM_ID': row["HADM_ID"], 'ICUSTAY_ID': row["ICUSTAY_ID"], 'ITEMID': itemid, \
                           'CHARTTIME': starttime, 'VALUENUM': dst_value, 'VALUEUOM': mainunit}]
            w.writerows(data_stats)
        ###############################################
        # case 2
        elif itemid in valid_lab_cate:
            if value is None or value == '':
                print('no value: ', row_no)
                continue
            value = catedict[itemid][value]
            if value is None:
                continue
            c += 1
            pbar.set_description("C-%s"%c)
            data_stats = [{'SUBJECT_ID': row["SUBJECT_ID"], 'HADM_ID': row["HADM_ID"], 'ICUSTAY_ID': row["ICUSTAY_ID"], 'ITEMID': itemid, \
                           'CHARTTIME': starttime, 'VALUENUM': value, 'VALUEUOM': valueuom}]
            w.writerows(data_stats)
        ###############################################
        # case 3
        elif itemid in valid_lab_num:
            if value is None or value == '':
                print('no value: ', row_no)
                continue
            ce2res = parseNum(value)
            if ce2res is None:
                print('not parsed')
                continue
            else:
                value = ce2res
            if valueuom is None:
                valueuom = ''
            currentunit = valueuom.replace(' ','').replace('<','').replace('>','').replace('=','').lower()
            mainunit = allitem_unit[map_itemid_index[itemid][0]]
            if currentunit == mainunit or currentunit == '':
                pass
            else:
                if itemid in unitmap.keys():
                    value = convert_units(unitmap[itemid], currentunit, mainunit, value)
                else:
                    if currentunit != mainunit:
                        value = None
            if value is None:
                print('not convertible: ', row_no)
                continue
            c += 1
            pbar.set_description("C-%s"%c)
            data_stats = [{'SUBJECT_ID': row["SUBJECT_ID"], 'HADM_ID': row["HADM_ID"], 'ICUSTAY_ID': row["ICUSTAY_ID"], 'ITEMID': itemid, \
                           'CHARTTIME': starttime, 'VALUENUM': value, 'VALUEUOM': valueuom}]
            w.writerows(data_stats)
        ###############################################
        # case 4
        elif itemid in valid_lab_ratio:
            if value is None or value == '':
                print('no value: ', row_no)
                continue
            try:
                fs = value.split('/')
                f1, f2 = fs[0], fs[1]
                if f1 != '':
                    value = float(f1)
                if f2 != '':
                    value = float(f2)
            except:
                print('not parsed: ', row_no)
                continue
            c += 1
            pbar.set_description("C-%s"%c)
            data_stats = [{'SUBJECT_ID': row["SUBJECT_ID"], 'HADM_ID': row["HADM_ID"], 'ICUSTAY_ID': row["ICUSTAY_ID"], 'ITEMID': itemid, \
                           'CHARTTIME': starttime, 'VALUENUM': value, 'VALUEUOM': valueuom}]
            w.writerows(data_stats)

    print("create ", table, " done!")
    ##########################################################
    ################       prescriptions        ###############
    ##########################################################
    # We only need to discard records without starttime or value.
    table = "prescriptions"
    nb_rows = nb_rows_dict[table.lower()]
    print("creating ", table)
    fn = os.path.join(output_path, table.upper()+".csv")
    if os.path.exists(fn) or os.path.isfile(fn):
        os.remove(fn)
    else:
        f = open(fn, 'w')
        f.write(','.join(obs_header) + '\n')
        f.close()
    w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
    #################################################################
    c = 0
    pbar = tqdm(read_events_table_by_row(mimic3_path, table.lower()),total=nb_rows)
    for row, row_no in pbar:
        starttime, itemid, value, valueuom = row["STARTDATE"], row["FORMULARY_DRUG_CD"],  row["DOSE_VAL_RX"], row["DOSE_UNIT_RX"]

        if pd.isnull(starttime) or pd.isnull(itemid):
            continue
        ###############################################
        # formatting the value
        dose = value
        dose = dose.replace(',', '').replace('<', '').replace('>', '').replace('=', '').replace(' ', '')
        numVal = None
        try:
            numVal = float(dose)
        except:
            if (len(dose.split('-')) == 2):
                strs = dose.split('-')
                try:
                    numVal = (float(strs[0]) + float(strs[1])) / 2.0
                except:
                    print('not parsed: ', pe)
                    continue
            else:
                print('not parsed: ', pe)
                continue

        # discard none value
        if (numVal == None):
            print('not parsed: ', row_no)
            continue

        # check unit
        # convert units...
        if valueuom is None:
            valueuom = ''
        valuenum = numVal
        valueuom = valueuom.replace(' ', '').lower()
        mainunit = allitem_unit[map_itemid_index[itemid][0]]
        if valueuom == mainunit or valueuom == '':
            dst_value = float(valuenum)
        else:
            dst_value = None

        # discard none value
        if (dst_value == None):
            print('not convertible: ', row_no)
            continue

        c += 1
        pbar.set_description("C-%s"%c)
        data_stats = [{'SUBJECT_ID': row["SUBJECT_ID"], 'HADM_ID': row["HADM_ID"], 'ICUSTAY_ID': row["ICUSTAY_ID"], \
                       'ITEMID': row["ITEMID"], 'CHARTTIME': row["CHARTTIME"],\
                       'VALUENUM': dst_value, 'VALUEUOM': valueuom}]
        w.writerows(data_stats)
    print("create ", table, " done!")



if __name__ == '__main__':
    main()
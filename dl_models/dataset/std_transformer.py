import os
import sys
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# standardizer for folds
class FoldsStandardizer(object):
    def __init__(self, serial_series, non_serial_series):
        self.serial_mean = serial_series[0]
        self.serial_std = serial_series[1]
        self.non_serial_mean = non_serial_series[0]
        self.non_serial_std = non_serial_series[1]

    def transform(self, X):
        print(X[0].shape)
        print(X[1].shape)
        assert len(X) == 2
        assert len(X[1].shape) == 3 # (id, time, feature)
        assert len(X[0].shape) == 2 # (id, feature)
        assert X[1].shape[2] == self.serial_mean.shape[0]
        assert X[1].shape[2] == self.serial_std.shape[0]
        assert X[0].shape[1] == self.non_serial_mean.shape[0]
        assert X[0].shape[1] == self.non_serial_std.shape[0]
        non_serial = np.copy(X[0])
        for id in xrange(non_serial.shape[0]):
            non_serial[id, :] = (non_serial[id, :] - self.non_serial_mean) / self.non_serial_std
        non_serial[np.isinf(non_serial)] = 0
        non_serial[np.isnan(non_serial)] = 0
        serial = np.copy(X[1])
        for id in xrange(serial.shape[0]):
            for t in xrange(serial.shape[1]):
                serial[id, t, :] = (serial[id, t, :] - self.serial_mean) / self.serial_std
        serial[np.isinf(serial)] = 0
        serial[np.isnan(serial)] = 0
        return [non_serial, serial]

class StaticFeaturesStandardizer(object):
    def __init__(self, train_mean, train_std):
        self.train_mean = train_mean
        self.train_std = train_std

    def transform(self, X):
        Xtrans = (X - self.train_mean) / self.train_std
        Xtrans[np.isinf(Xtrans)] = 0.0
        Xtrans[np.isnan(Xtrans)] = 0.0
        return Xtrans

class SAPSIITransformer(object):
    def __init__(self, train_idx):
        self.train_idx = train_idx

    def transform(self, X):
        '''
        [('GCS', 0)], 'mengcz_vital_ts': [('SysBP_Mean', 1), ('HeartRate_Mean', 2), ('TempC_Mean', 3)],
        'mengcz_pao2fio2_ts': [('PO2', 4), ('FIO2', 5)], 'mengcz_urine_output_ts': [('UrineOutput', 6)],
        'mengcz_labs_ts': [('BUN_min', 7), ('WBC_min', 8), ('BICARBONATE_min', 9), ('SODIUM_min', 10),
        ('POTASSIUM_min', 11), ('BILIRUBIN_min', 12)]

        age: 0, aids: 1, he,: 2, mets: 3, admissiontype: 4
        '''
        non_serial = np.copy(X[0])
        serial = np.copy(X[1])

        for admid in range(non_serial.shape[0]):
            # non_serial
            age, aids, hem, mets, admissiontype = non_serial[admid][0], non_serial[admid][1], non_serial[admid][2], non_serial[admid][3], non_serial[admid][4]

            try:
                age = age / 365.25
                if age < 40:
                    non_serial[admid][0] = 0.0
                elif age < 60:
                    non_serial[admid][0] = 7.0
                elif age < 70:
                    non_serial[admid][0] = 12.0
                elif age < 75:
                    non_serial[admid][0] = 15.0
                elif age < 80:
                    non_serial[admid][0] = 16.0
                elif age >= 80:
                    non_serial[admid][0] = 18.0
            except:
                non_serial[0] = 0.0

            try:
                if aids == 1:
                    non_serial[admid][1] = 17.0
                else:
                    non_serial[admid][1] = 0.0
            except:
                non_serial[admid][1] = 0.0

            try:
                if hem == 1:
                    non_serial[admid][2] = 10.0
                else:
                    non_serial[admid][2] = 0.0
            except:
                non_serial[admid][2] = 0.0

            try:
                if mets == 1:
                    non_serial[admid][3] = 9.0
                else:
                    non_serial[admid][3] = 0.0
            except:
                non_serial[admid][3] = 0.0

            try:
                if admissiontype == 0: # medical
                    non_serial[admid][4] = 6.0
                elif admissiontype == 1: # sche
                    non_serial[admid][4] = 0.0
                elif admissiontype == 2: # unsche
                    non_serial[admid][4] = 8.0
            except:
                non_serial[admid][4] = 0.0

            # serial
            for t in range(serial[admid].shape[0]):
                gcs = serial[admid][t][0]
                sbp = serial[admid][t][1]
                hr = serial[admid][t][2]
                bt = serial[admid][t][3]
                pfr = serial[admid][t][4]
                uo = serial[admid][t][5]
                sunl = serial[admid][t][6]
                wbc = serial[admid][t][7]
                sbl = serial[admid][t][8]
                sl = serial[admid][t][9]
                pl = serial[admid][t][10]
                bl = serial[admid][t][11]

                try:
                    if hr < 40:
                        serial[admid][t][2] = 11.0
                    elif hr >= 160:
                        serial[admid][t][2] = 7.0
                    elif hr >= 120:
                        serial[admid][t][2] = 4.0
                    elif hr < 70:
                        serial[admid][t][2] = 2.0
                    elif hr >= 70 and hr < 120:
                        serial[admid][t][2] = 0.0
                    else:
                        serial[admid][t][2] = 0.0
                except:
                    serial[admid][t][2] = 0.0

                try:
                    if sbp < 70:
                        serial[admid][t][1] = 13.0
                    elif sbp < 100:
                        serial[admid][t][1] = 5.0
                    elif sbp >= 200:
                        serial[admid][t][1] = 2.0
                    elif sbp >= 100 and sbp < 200:
                        serial[admid][t][1] = 0.0
                    else:
                        serial[admid][t][1] = 0.0
                except:
                    serial[admid][t][1] = 0.0

                try:
                    if bt < 39.0:
                        serial[admid][t][3] = 0.0
                    elif bt >= 39.0:
                        serial[admid][t][3] = 3.0
                    else:
                        serial[admid][t][3] = 0.0
                except:
                    serial[admid][t][3] = 0.0

                try:
                    if pfr < 100:
                        serial[admid][t][4] = 11.0
                    elif pfr < 200:
                        serial[admid][t][4] = 9.0
                    elif pfr >= 200:
                        serial[admid][t][4] = 6.0
                    else:
                        serial[admid][t][4] = 0.0
                except:
                    serial[admid][t][4] = 0.0

                try:
                    if uo < 500:
                        serial[admid][t][5] = 11.0
                    elif uo < 1000:
                        serial[admid][t][5] = 4.0
                    elif uo >= 1000:
                        serial[admid][t][5] = 0.0
                    else:
                        serial[admid][t][5] = 0.0
                except:
                    serial[admid][t][5] = 0.0

                try:
                    if sunl < 28.0:
                        serial[admid][t][6] = 0.0
                    elif sunl < 83.0:
                        serial[admid][t][6] = 6.0
                    elif sunl >= 84.0:
                        serial[admid][t][6] = 10.0
                    else:
                        serial[admid][t][6] = 0.0
                except:
                    serial[admid][t][6] = 0.0

                try:
                    if wbc < 1.0:
                        serial[admid][t][7] = 12.0
                    elif wbc >= 20.0:
                        serial[admid][t][7] = 3.0
                    elif wbc >= 1.0 and wbc < 20.0:
                        serial[admid][t][7] = 0.0
                    else:
                        serial[admid][t][7] = 0.0
                except:
                    serial[admid][t][7] = 0.0

                try:
                    if pl < 3.0:
                        serial[admid][t][10] = 3.0
                    elif pl >= 5.0:
                        serial[admid][t][10] = 3.0
                    elif pl >= 3.0 and pl < 5.0:
                        serial[admid][t][10] = 0.0
                    else:
                        serial[admid][t][10] = 0.0
                except:
                    serial[admid][t][10] = 0.0

                try:
                    if sl < 125:
                        serial[admid][t][9] = 5.0
                    elif sl >= 145:
                        serial[admid][t][9] = 1.0
                    elif sl >= 125 and sl < 145:
                        serial[admid][t][9] = 0.0
                    else:
                        serial[admid][t][9] = 0.0
                except:
                    serial[admid][t][9] = 0.0

                try:
                    if sbl < 15.0:
                        serial[admid][t][8] = 5.0
                    elif sbl < 20.0:
                        serial[admid][t][8] = 3.0
                    elif sbl >= 20.0:
                        serial[admid][t][8] = 0.0
                    else:
                        serial[admid][t][8] = 0.0
                except:
                    serial[admid][t][8] = 0.0

                try:
                    if bl < 4.0:
                        serial[admid][t][11] = 0.0
                    elif bl < 6.0:
                        serial[admid][t][11] = 4.0
                    elif bl >= 6.0:
                        serial[admid][t][11] = 9.0
                    else:
                        serial[admid][t][11] = 0.0
                except:
                    serial[admid][t][11] = 0.0

                try:
                    if gcs < 3:
                        serial[admid][t][0] = 0.0
                    elif gcs < 6:
                        serial[admid][t][0] = 26.0
                    elif gcs < 9:
                        serial[admid][t][0] = 13.0
                    elif gcs < 11:
                        serial[admid][t][0] = 7.0
                    elif gcs < 14:
                        serial[admid][t][0] = 5.0
                    elif gcs >= 14 and gcs <= 15:
                        serial[admid][t][0] = 0.0
                    else:
                        serial[admid][t][0] = 0.0
                except:
                    serial[admid][t][0] = 0.0
        non_serial_mean, non_serial_std = np.nanmean(non_serial[self.train_idx], axis=0), np.nanstd(non_serial[self.train_idx], axis=0)
        non_serial = (non_serial - non_serial_mean) / non_serial_std
        non_serial[np.isnan(non_serial)] = 0.0
        non_serial[np.isinf(non_serial)] = 0.0

        serial_mean, serial_std = np.nanmean(np.concatenate(serial[self.train_idx], axis=0), axis=0), np.nanstd(np.concatenate(serial[self.train_idx], axis=0), axis=0)
        serial = (serial - serial_mean) / serial_std
        serial[np.isnan(serial)] = 0.0
        serial[np.isinf(serial)] = 0.0

        return [non_serial, serial]

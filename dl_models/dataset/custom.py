import os, random
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
"""
def list_with_index(X, idx=None):
    if type(X) is np.ndarray:
        if idx is None:
            return X
        else:
            return X[idx]
    return [list_with_index(x, idx) for x in X]

class standDataset(Dataset):
    def __init__(self, data_file_pathname, idxs, tsf, label_type, task_name):
        data_file = np.load(data_file_pathname)
        ###############################################
        self.X_s = data_file['adm_features_all']
        self.X_t = data_file['ep_tdata']
        if model_type == 1:
            self.X =[self.X_s, self.X_t]
        else:
            self.X =self.X_s
        ###############################################
        # Set tasks
        if task_name == 'icd9':
            self.y = data_file['y_icd9'][:,label_type]
            self.y = (self.y > 0).astype("float")
        elif task_name == 'mor':
            adm_labels = data_file['adm_labels_all']
            self.y = adm_labels[:, label_type]
            self.y = (self.y > 0).astype("float")
        elif task_name == 'los':
            # convert minute to hour
            self.y = data_file['y_los'] / 60.0
        ###############################################
        if tsf is not None:
            self.X = tsf.fit(
                list_with_index(self.X, idxs),
                y_fitting[idxs]
            ).transform(self.X)
        else:
            # No transform at all
            self.X = self.X
        self.y = self.y[idxs]
        ###############################################
        self.len = len(self.X_s)

    def __getitem__(self, index):
        X_s = torch.Tensor(self.X_s[index])
        X_t = torch.Tensor(self.X_t[index])
        Y = torch.Tensor(self.y[index])
        return X_s, X_t, Y

    def __len__(self):
        return self.len
"""

def callback_get_label(dataset, idx):
    return int(dataset[idx][-1])

class customDataset(Dataset):
    def __init__(self, data_file_pathname, idxs, label_type, task_name, tranformer):
        data_file = np.load(data_file_pathname)
        ###############################################
        self.X_s = data_file['adm_features_all']
        self.X_t = data_file['ep_tdata']
        ###############################################
        # Set tasks
        if task_name == 'icd9':
            self.y = data_file['y_icd9'][:, label_type]
            self.y = (self.y > 0).astype("float")
        elif task_name == 'mor':
            adm_labels = data_file['adm_labels_all']
            self.y = adm_labels[:, label_type]
            self.y = (self.y > 0).astype("float")
        elif task_name == 'los':
            # convert minute to hour
            self.y = data_file['y_los'] / 60.0
            # convert minute to years
            #self.y = data_file['y_los'] / 60.0 / 24.0 / 365.0
        ###############################################
        # Get subset
        self.X_s = self.X_s[idxs]
        self.X_t = self.X_t[idxs]
        self.y = self.y[idxs]
        ###############################################
        # Standardization
        if tranformer is not None:
            self.X_s, self.X_t = tranformer.transform([self.X_s, self.X_t])
        ###############################################
        self.len = len(self.y)

    def __getitem__(self, index):
        X_s = torch.Tensor(self.X_s[index])
        X_t = torch.Tensor(self.X_t[index])
        Y = torch.Tensor([self.y[index]])
        return X_s, X_t, Y

    def __len__(self):
        return self.len

class staticDataset(Dataset):
    def __init__(self, data_file_pathname, static_features_path, idxs, label_type, task_name, tranformer):
        data_file = np.load(data_file_pathname)
        data_file2 = np.load(static_features_path)
        ###############################################
        self.X_s = data_file2["hrs_mean_array"]
        ###############################################
        # Set tasks
        if task_name == 'icd9':
            self.y = data_file['y_icd9'][:, label_type]
            self.y = (self.y > 0).astype("float")
        elif task_name == 'mor':
            adm_labels = data_file['adm_labels_all']
            self.y = adm_labels[:, label_type]
            self.y = (self.y > 0).astype("float")
        elif task_name == 'los':
            # convert minute to hour
            self.y = data_file['y_los'] / 60.0
            # convert minute to years
            #self.y = data_file['y_los'] / 60.0 / 24.0 / 365.0
        ###############################################
        # Get subset
        self.X_s = self.X_s[idxs]
        self.y = self.y[idxs]
        ###############################################
        # Standardization
        if tranformer is not None:
            self.X_s = tranformer.transform(self.X_s)
        ###############################################
        self.len = len(self.y)

    def __getitem__(self, index):
        X_s = torch.Tensor(self.X_s[index])
        Y = torch.Tensor([self.y[index]])
        return X_s, Y

    def __len__(self):
        return self.len

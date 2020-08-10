import os, random
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class customDataset(Dataset):
    # 'icd9'
    def __init__(self, data_file_pathname, idxs, tsf, task_id, task_name, model_type):
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
            self.y = data_file['y_icd9'][:,task_id]
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
        return X_s, Y

    def __len__(self):
        return self.len


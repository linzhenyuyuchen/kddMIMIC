import os, yaml, torch
import numpy as np
import pandas as pd
from glob import glob
import SimpleITK as sitk
from torch.utils.data import Dataset

class nnDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(self.y)

    def __getitem__(self, index):
        X = torch.Tensor(self.X[index])
        Y = torch.Tensor([self.y[index]])
        return X, Y

    def __len__(self):
        return self.len

class mDataset():
    def __init__(self, feature_path, class_names, feature_names = None):
        self.feature_path = feature_path
        self.class_names = class_names
        self.feature_names = feature_names

    def getDataset(self):
        X = []
        y = []
        for cname in self.class_names.keys():
            feature = pd.read_csv(self.feature_path + "/" + cname + ".csv", low_memory=False)
            if self.feature_names:
                feature = feature[self.feature_names]

            feature = feature.values.tolist()
            print("get #", len(feature), " of ", cname)
            label = [self.class_names[cname]] * len(feature)
            X.extend(feature)
            y.extend(label)

        return np.array(X), np.array(y)

class img3dDataset():
    def __init__(self, resized_dir, class_names, feature_names = None):
        self.resized_dir = resized_dir
        self.class_names = class_names
        self.feature_names = feature_names

    def get_array_from_itk(self, path):
        filePath = self.resized_dir + path
        itk_img = sitk.ReadImage(filePath)
        img_array = sitk.GetArrayFromImage(itk_img)
        return img_array

    def get_array(self, path):
        return np.load(path)

    def concateMask(self, image_path, mask_path):
        img = self.get_array(image_path)
        mask = self.get_array(mask_path)
        con = np.concatenate((img, mask))
        return con[np.newaxis, :]

    def getDataset(self):
        X = []
        y = []
        for cname in self.class_names.keys():
            fs = glob(self.resized_dir + cname + "/Mask-*.npy")

            feature = [self.concateMask(f.replace("Mask-", ""), f) for f in fs]
            print("get #", len(fs), " of ", cname)
            lb = [0, 0, 0]
            lb[int(self.class_names[cname])] = 1
            label = [lb] * len(feature)
            X.extend(feature)
            y.extend(label)
        # [n, 1, depth, width, height]
        return np.array(X), np.array(y)

# if __name__ == '__main__':

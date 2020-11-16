import numpy as np
from sklearn import feature_selection
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

class featurePrepro():
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test

    def simpleImpute(self, strategy='mean'):
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(self.X_train)
        self.X_train = imp.transform(self.X_train)
        self.X_test = imp.transform(self.X_test)
        
    def standardScale(self):
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def normalizer(self):
        nor = Normalizer()
        nor.fit(self.X_train)
        self.X_train = nor.transform(self.X_train)
        self.X_test = nor.transform(self.X_test)

    def minMaxScaler(self):
        mms = MinMaxScaler()
        mms.fit(self.X_train)
        self.X_train = mms.transform(self.X_train)
        self.X_test = mms.transform(self.X_test)
    
    def pca(self, n_components=10):
        pca = PCA(n_components=n_components)
        pca.fit(self.X_train)
        self.X_train = pca.transform(self.X_train)
        self.X_test = pca.transform(self.X_test)

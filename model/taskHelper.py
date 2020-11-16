import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold, train_test_split
################################################
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
################################################
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
################################################
from .nnmodels.ffn import FeedForwardNetwork
from .botHelper import BaseBot

class taskHelper():
    def __init__(self, model_name="rf", type_name="cls", nfold=5):
        self.model_name = model_name
        self.type_name = type_name
        self.nfold = nfold

    def getModel(self):
        print("selected model: ", self.model_name)
        if self.type_name =="cls":
            if self.model_name == 'rf':
                model = RandomForestClassifier(n_estimators=10, max_depth=7, max_features='sqrt', random_state=0)
            elif self.model_name == 'svm':
                model = SVC(gamma='auto')
            elif self.model_name == 'ada':
                model = AdaBoostClassifier(n_estimators=100)
            elif self.model_name == 'ridge':
                model = RidgeClassifier()
            elif self.model_name == 'kn':
                model = KNeighborsClassifier(n_neighbors=8)
            else:
                model = MLPClassifier(max_iter=2000)
        else:
            if self.model_name == 'log':
                model = LogisticRegression()
            elif self.model_name == 'lasso':
                model = Lasso()
            elif self.model_name == 'elastic':
                model = ElasticNet()
            else:
                model = BaggingRegressor(n_estimators=10)

        return model

    def splitFold(self, x, y):
        kf = StratifiedShuffleSplit(n_splits=self.nfold, test_size=0.2, random_state=0)
        idx_trva_list = []
        idx_te_list = []
        for idx_tr, idx_te in kf.split(X, y):
            idx_trva_list.append(idx_tr)
            idx_te_list.append(idx_te)
        idx_list = np.empty([self.nfold, 3], dtype=object)
        for i in range(self.nfold):
            idx_list[i][0] = np.setdiff1d(idx_trva_list[i], idx_te_list[(i + 1) % self.nfold], True)
            idx_list[i][1] = idx_te_list[(i + 1) % self.nfold]
            idx_list[i][2] = idx_te_list[i]
        return idx_list

    def evaluation(self, pred, gt):
        if self.type_name =="cls":
            print("Accuracy", metrics.accuracy_score(gt, pred))
        else:
            error = metrics.mean_squared_error(gt, pred)
            error_mae = metrics.mean_absolute_error(gt, pred)
            print("MSE :", error)
            print("MAE :", error_mae)

    def train_cv(self, X, y):
        print("starting CV-train..")
        idx_list = self.splitFold(X, y)
        pred_y_all = []
        global_y_all = []
        n_fold = 0
        for idx_tr, idx_va, idx_te in idx_list:
            # Build dataset
            n_fold += 1
            idx_trva = np.concatenate([idx_tr, idx_va])
            X_trva, y_trva = X[idx_trva], y[idx_trva]
            # X_tr, y_tr = X[idx_tr], y[idx_tr]
            # X_va, y_va = X[idx_va], y[idx_va]
            X_te, y_te = X[idx_te], y[idx_te]
            # Train the model
            clf = self.getModel()
            clf.fit(X_trva, y_trva)
            y_pred = clf.predict(X_te)
            pred_y_all.extend(y_pred.tolist())
            global_y_all.extend(y_te.tolist())

        self.evaluation(pred_y_all, global_y_all)

    def train(self, X, y):
        print("starting train..")
        # Train the model
        self.clf = self.getModel()
        self.clf.fit(X, y)

    def test(self, X, y):
        y_pred = self.clf.predict(X)
        self.evaluation(y_pred, y)


class nntaskHelper():
    def __init__(self, n_features=10, num_labels=1):
        self.n_features = n_features
        self.num_labels = num_labels

    def getModel(self):
        model = FeedForwardNetwork(n_features=self.n_features, hidden_dim=self.n_features//2, y_tasks=self.num_labels)
        return model

    def evaluation(self, pred, gt):
        if self.num_labels > 1:
            print("Accuracy", metrics.accuracy_score(gt, pred))
        else:
            error = metrics.mean_squared_error(gt, pred)
            error_mae = metrics.mean_absolute_error(gt, pred)
            print("MSE :", error)
            print("MAE :", error_mae)


    def train(self, cfg, train_dataset, val_dataset, test_dataset):
        print("starting train..")
        # Train the model
        nb_epoch = cfg["nb_epoch"]
        learning_rate = cfg["learning_rate"]
        early_stopping_patience = cfg["early_stopping_patience"]
        train_batch_size = cfg["train_batch_size"]
        val_batch_size = cfg["val_batch_size"]
        test_batch_size = cfg["test_batch_size"]
        #################################################################
        train_sampler = RandomSampler(train_dataset)
        # train_sampler = ImbalancedDatasetSampler(train_dataset, callback_get_label=callback_get_label)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size,num_workers=4, drop_last=True)
        val_sampler = RandomSampler(val_dataset)
        # val_sampler = ImbalancedDatasetSampler(val_dataset, callback_get_label=callback_get_label)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=val_batch_size, num_workers=4)
        test_sampler = RandomSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size, num_workers=4)
        #################################################################
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.getModel()
        model.to(device)
        #################################################################
        # optimizer
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        #################################################################
        # train
        bot = BaseBot(model, train_dataloader, val_dataloader, optimizer, patience=early_stopping_patience)
        if self.num_labels==1:
            bot.set_label_type(1)
            bot.set_loss_function("MSELoss")
        bot.train_ffn(n_epoch=nb_epoch)
        bot.save_model()
        y_pred_tr, y_tr = bot.predict_ffn(train_dataloader)
        y_pred_va, y_va = bot.predict_ffn(val_dataloader)
        y_pred, y_te = bot.predict_ffn(test_dataloader)
        
        if self.num_labels==1:
            print("MSE on testset:", bot.mse(y_pred, y_te))
            print("MAE on testset:", bot.mae(y_pred, y_te))
        else:
            print("mean accuracy on trainset:", bot.calAccuracy(y_pred_tr, y_tr))
            print("mean accuracy on   valset:", bot.calAccuracy(y_pred_va, y_va))
            print("mean accuracy on  testset:", bot.calAccuracy(y_pred, y_te))


import yaml
import pickle
import random
import numpy as np
from matplotlib import pylab as plt
import plotly
import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
#plotly.offline.init_notebook_mode(connected=True)

#from model.taskHelper import taskHelper
from preprocessing.dataset import mDataset
from preprocessing.featureSelector import featureSelector

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error

if __name__ == "__main__":
    # Hyper-parameters
    cfg = yaml.load(open("./cfg.yaml", "r"), Loader=yaml.FullLoader)
    class_names = cfg["class_names"]
    feature_names = cfg["feature_names"]
    feature_dir = cfg["feature_dir"]


    # Configure
    show_feature_name = False
    feature_dir1 = "/data1/lzy/3st/features_delta_exp_1mm/"
    feature_dir2 = "/data1/lzy/3st/features_delta_shr_1mm/"
    class_names_train = {
        "C058_train_origin_delta": 0,
        "C058_train_exp1_delta": 1,
        "C058_train_shr1_delta": -1,
        "C058_train_exp2_delta": 2,
        "C058_train_shr2_delta": -2,
        # "C058_train_exp3_delta": 3,
        # "C058_train_shr3_delta": -3,
        # "C058_train_exp4_delta": 4,
        # "C058_train_shr4_delta": -4,
    }
    class_names_test = {
        "C058_test_origin_delta": 0,
        "C058_test_exp1_delta": 1,
        "C058_test_shr1_delta": -1,
        "C058_test_exp2_delta": 2,
        "C058_test_shr2_delta": -2,
        # "C058_test_exp3_delta": 3,
        # "C058_test_shr3_delta": -3,
        # "C058_test_exp4_delta": 4,
        # "C058_test_shr4_delta": -4,
    }

    myDataset = mDataset(feature_dir1, class_names_train, feature_names)
    X_train1, y_train1 = myDataset.getDataset()
    myDataset = mDataset(feature_dir2, class_names_train, feature_names)
    X_train2, y_train2 = myDataset.getDataset()

    myDataset = mDataset(feature_dir1, class_names_test, feature_names)
    X_test1, y_test1 = myDataset.getDataset()
    myDataset = mDataset(feature_dir2, class_names_test, feature_names)
    X_test2, y_test2 = myDataset.getDataset()

    X_train = np.concatenate([X_train1,X_train2],axis=1)
    y_train = y_train1
    X_test = np.concatenate([X_test1,X_test2],axis=1)
    y_test = y_test1

    print(X_train.shape)
    print(X_test.shape)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)


    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    nor = Normalizer()
    nor.fit(X_train)
    X_train = nor.transform(X_train)
    X_test = nor.transform(X_test)

    mms = MinMaxScaler()
    mms.fit(X_train)
    X_train = mms.transform(X_train)
    X_test = mms.transform(X_test)

    feature_name_all = []

    model2 = BaggingRegressor(n_estimators=30)
    model2.fit(X_train, y_train)
    guesses = model2.predict(X_test)
    error = mean_squared_error(y_test, guesses)
    error_mae = mean_absolute_error(y_test, guesses)
    print("BaggingRegressor MSE :", error)
    print("BaggingRegressor MAE :", error_mae)

    dists = [4,3,2,1,0,-1,-2,-3,-4]
    dists = [2,1,0,-1,-2]
    c1 = ["red","maroon", "yellow","green","purple","peru","cyan","pink","slategray","black", "orange"]
    lbs = []

    fig = go.Figure()

    for d in range(len(dists)):
        dist = dists[d]
        if dist==0:
            a="Normal"
        if dist<0:
            a=f"Shrink--{-1*dist} mm"
        if dist>0:
            a=f"Expand--{dist} mm"
        c = 0
        y_axis1 = []
        y_axis2 = []
        for i in range(len(y_test)):

            if y_test[i]==dist:
                y_axis1.append(guesses[i])
                y_axis2.append(y_test[i])
        #ln1, = plt.plot(list(range(len(y_axis1))), y_axis1, color="red")
        random.shuffle(y_axis1)
        trace = go.Scatter(x=list(range(len(y_axis1))), y=y_axis1, mode='markers', name=f"{a}")
        fig.add_trace(trace)
        #lbs.append(trace)
    fig.update_layout(
        title="Regression of distance",
        xaxis_title="Samples",
        yaxis_title="Predicted value",
        font=dict(
            #family="Courier New, monospace",
            size=13,
            #color="RebeccaPurple"
        )
    )
    fig.show()

    #py.plot(lbs)
    #pio.write_image(fig,"plotly.png")


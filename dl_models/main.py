import os, yaml, pickle, argparse
import random, logging, json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

from model.cls_model import *
from dataset.custom import *
from dataset.std_transformer import *
from utils.bot_helper import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# handler = logging.FileHandler("log.txt")
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
#
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# logger.addHandler(console)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if os.environ.get("DETERMINISTIC", None):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(12321)

def get_arg():
    #### Start here for the common configs ####
    arg_parser = argparse.ArgumentParser()
    # basic
    arg_parser.add_argument('data_name', type=str, default="mor")
    arg_parser.add_argument('task_name', type=str, default="mor")
    arg_parser.add_argument('model_type', type=int, default=1)
    arg_parser.add_argument('--data_file_name', type=str, default="/data3/Benchmarking_DL_MIMICIII/Data/admdata_99p/24hrs_raw/series/imputed-normed-ep_1_24.npz")
    # arg_parser.add_argument('--folds_file_name', type=str, default="..")
    # arg_parser.add_argument('--folds_stat_file_name', type=str, default="..")
    arg_parser.add_argument('--static_features_path', type=str, default="/data3/Benchmarking_DL_MIMICIII/Data/admdata_99p/24hrs_raw/non_series/tsmean_24hrs.npz")
    arg_parser.add_argument('--label_type', type=int, default=0)
    arg_parser.add_argument('--working_path', '-p', type=str, default='../')
    arg_parser.add_argument('--random_seed', type=int, default=12321)
    # training
    arg_parser.add_argument('--batch_size', type=int, default=100)
    arg_parser.add_argument('--nb_epoch', type=int, default=250)
    arg_parser.add_argument('--batch_normalization', type=str, default='True')
    arg_parser.add_argument('--learning_rate', type=float, default=0.001)
    # models
    arg_parser.add_argument('--without_static', action='store_true')
    arg_parser.add_argument('--ffn_depth', type=int, default=1)
    arg_parser.add_argument('--merge_depth', type=int, default=0)
    arg_parser.add_argument('--output_dim', type=int, default=2)
    arg_parser.add_argument('--n_features', type=int, default=136)
    arg_parser.add_argument('--time_step', type=int, default=48)
    arg_parser.add_argument('--dropout', type=float, default=0.1)
    arg_parser.add_argument('--static_ffn_depth', type=int, default=2)
    arg_parser.add_argument('--static_hidden_dim', type=int, default=2048)

    return arg_parser.parse_args()

def get_model(args):
    task_name = args.task_name
    model_type = args.model_type
    fit_parameters = [args.output_dim, args.ffn_depth, args.merge_depth]
    batch_normalization = args.batch_normalization
    dropout = args.dropout
    without_static = args.without_static
    static_ffn_depth = args.static_ffn_depth
    static_hidden_dim = args.static_hidden_dim
    ##############################################
    time_step = args.time_step
    n_features = args.n_features
    ##############################################
    # Set tasks
    if task_name == 'icd9':
        y_tasks = 2
    elif task_name == 'mor':
        y_tasks = 2
    elif task_name == 'los':
        y_tasks = 1
    # Set model
    if args.model_type == 1:
        # build model
        logger.info(f"model type: HMM, without_static: {without_static}, time_step: {time_step}")
        model = HierarchicalMultimodal(static = not without_static, size_Xs= 5, dropout = dropout, batch_normalization = batch_normalization,
                                       time_step = time_step, n_features = n_features, fit_parameters = fit_parameters, y_tasks = y_tasks)
    elif args.model_type == 2:
        # build model
        logger.info(f"model type: FFN")
        model = FeedForwardNetwork(n_features=n_features, hidden_dim=static_hidden_dim, y_tasks = y_tasks, ffn_depth=static_ffn_depth, batch_normalization=batch_normalization)

    return model

def mse(pred, y):
    pred = np.array(pred)
    y = np.array(y)
    mse = metrics.mean_squared_error(y, pred)
    return mse

def metric_auroc_auprc(pred, y):
    pred = np.array(pred)
    y = np.array(y)

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    auroc_score = metrics.auc(fpr, tpr)

    auprc_score = metrics.average_precision_score(y, pred)

    return auroc_score, auprc_score

def run_folds(args):
    task_name = args.task_name
    data_file_pathname = args.data_file_name
    label_type = args.label_type
    model_type = args.model_type
    static_features_path = args.static_features_path
    if without_static:
        model_type = str(model_type) + "_without_static"
    else:
        model_type = str(model_type)
    result_score_path = os.path.join(args.working_path, 'output', args.data_name,
                               "model" + model_type, "labeltype" + str(args.label_type),
                               "result_score.txt")
    #############################################################
    # load data
    data_file = np.load(data_file_pathname)
    X_s = data_file['adm_features_all']
    X_t = data_file['ep_tdata']
    if task_name == 'icd9':
        y = data_file['y_icd9'][:, label_type]
        y = (y > 0).astype("float")
    elif task_name == 'mor':
        y = data_file['adm_labels_all'][:, label_type]
        y = (y > 0).astype("float")
    elif task_name == 'los':
        y = data_file['adm_labels_all'][:, 0]
        y = (y > 0).astype("float")
    #############################################################
    # make folds
    logger.info(f"making folds..")
    kf = StratifiedKFold(n_splits=5)
    n_fold = 0
    pred_y_all = []
    global_y_all = []
    if args.model_type == 1:
        for idx_trva, idx_te in kf.split(X_t, y):
            # Build Dataset
            n_fold += 1
            logger.info(f"loading dataset of fold - {n_fold}..")
            stats, nsstats = get_standardize_stats_for_training(X_t[idx_trva], X_s[idx_trva])
            tranformer = FoldsStandardizer(stats, nsstats)
            train_dataset = customDataset(data_file_pathname, idx_trva, label_type, task_name, tranformer)
            dev_dataset = customDataset(data_file_pathname, idx_te, label_type, task_name, tranformer)
            # Train the model
            pred_y, global_y = train(args, n_fold, train_dataset, dev_dataset)
            [pred_y_all.append(o) for o in pred_y]
            [global_y_all.append(o) for o in global_y]
    else:
        data_file2 = np.load(static_features_path)
        X_s = data_file2["hrs_mean_array"]
        for idx_trva, idx_te in kf.split(X_s, y):
            # Build Dataset
            n_fold += 1
            logger.info(f"loading dataset of fold - {n_fold}..")
            tmean = np.nanmean(X_s[idx_trva], axis=0)
            tstd = np.nanstd(X_s[idx_trva], axis=0)
            tranformer = StaticFeaturesStandardizer(tmean, tstd)
            train_dataset = staticDataset(data_file_pathname, static_features_path, idx_trva, label_type, task_name, tranformer)
            dev_dataset = staticDataset(data_file_pathname, static_features_path, idx_te, label_type, task_name, tranformer)
            # Train the model
            pred_y, global_y = train(args, n_fold, train_dataset, dev_dataset)
            [pred_y_all.append(o) for o in pred_y]
            [global_y_all.append(o) for o in global_y]
    if task_name == 'los':
        mse_score = mse(pred_y_all, global_y_all)
        logger.info("=" * 25 + "MSE %f" + "=" * 25, mse_score)
        with open(result_score_path, "w") as f:
            f.write("MSE: " + str(mse_score))
            f.close()
    else:
        auroc_score, auprc_score = metric_auroc_auprc(pred_y_all, global_y_all)
        logger.info("=" * 25 + "AUROC %f" + "=" * 25, auroc_score)
        logger.info("=" * 25 + "AUPRC %f" + "=" * 25, auprc_score)
        with open(result_score_path, "w") as f:
            f.write("AUROC: " + str(auroc_score) + " / AUPRC: " + str(auprc_score))
            f.close()
    #############################################################


def train(args, n_fold, train_dataset, test_dataset):
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    nb_epoch = args.nb_epoch
    model_type = args.model_type
    ##############################################
    # Settings for task, model, path, etc
    if without_static:
        model_type = str(model_type) + "_without_static"
    else:
        model_type = str(model_type)
    result_path = os.path.join(args.working_path, 'output', args.data_name,
                               "model"+model_type, "labeltype"+str(args.label_type), "fold_"+str(n_fold))
    result_log_path = os.path.join(result_path, 'log')
    model_path = os.path.join(result_path, 'model')
    for required_path in [result_path, result_log_path, model_path]:
        if not os.path.exists(required_path):
            os.makedirs(required_path)
    logger.info(f"output dir: {result_path}")
    #################################################################
    train_batch_size = batch_size
    test_batch_size = batch_size
    #################################################################
    # dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size,num_workers=4)
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size,num_workers=4)
    ##############################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args)
    model.to(device)
    #################################################################
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #################################################################
    # train
    logger.info(f"training..")
    bot = BaseBot(model, train_dataloader, test_dataloader, optimizer,
                 log_dir=result_log_path, log_level=logging.INFO,
                 checkpoint_dir=model_path, echo=False,
                 device=device, use_tensorboard=False, use_amp=False, seed=123, n_gpus=1)
    if args.task_name == "los":
        bot.set_label_type(1)
        bot.set_loss_function("MSELoss")
    if args.model_type == 1:
        bot.train(n_epoch=nb_epoch)
        # save model
        bot.save_model()
        # test
        pred_y, global_y = bot.predict(test_dataloader)
    else:
        bot.train_ffn(n_epoch=nb_epoch)
        # save model
        bot.save_model()
        # test
        pred_y, global_y = bot.predict_ffn(test_dataloader)
    return pred_y, global_y

def main():
    ##############################################
    args = get_arg()
    run_folds(args)

if __name__ == "__main__":
    main()

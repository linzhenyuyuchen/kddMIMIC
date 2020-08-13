import os, yaml, pickle, argparse
import random, logging, json
import numpy as np
import pandas as pd
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if os.environ.get("DETERMINISTIC", None):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_arg():
    #### Start here for the common configs ####
    arg_parser = argparse.ArgumentParser()
    # basic
    arg_parser.add_argument('data_name', type=str, default="mor")
    arg_parser.add_argument('task_name', type=str, default="mor")
    arg_parser.add_argument('model_type', type=int, default=1)
    arg_parser.add_argument('--data_file_name', type=str, default="/data3/Data/admdata_99p/24hrs_raw/series/imputed-normed-ep_1_24.npz")
    # arg_parser.add_argument('--folds_file_name', type=str, default="..")
    # arg_parser.add_argument('--folds_stat_file_name', type=str, default="..")
    # arg_parser.add_argument('--static_features_path', type=str, default="..")
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

def run_folds(args, model):
    task_name = args.task_name
    data_file_pathname = args.data_file_name
    label_type = args.label_type
    #############################################################
    # n_reps = min(len(splits), 5)
    # for rep in range(n_reps):
    #     fold_idxs = splits[rep]
    #     # n folds in each repeat
    #     for i_fold, fold_idx in enumerate(fold_idxs):
    #         # train/validation/test set or train/test set
    #         if len(fold_idx) == 2:
    #             idx_trva, idx_te = fold_idx
    #         elif len(fold_idx) == 3:
    #             idx_tr, idx_va, idx_te = fold_idx
    #             idx_trva = np.concatenate((idx_tr, idx_va))
    #         # Build Dataset
    #         train_dataset = customDataset(data_file_pathname, idx_trva, tsf, label_type, task_name, model_type)
    #         dev_dataset = customDataset(data_file_pathname, idx_te, tsf, label_type, task_name, model_type)
    #         # Train the model
    #         train(args, train_dataset, dev_dataset, model)
    #############################################################
    # make folds
    logger.info(f"making folds..")
    idxs = np.array(list(range(16993)))
    idx_te = np.random.choice(idxs, size=4248, replace=False)
    idx_trva = list(set(idxs) - set(idx_te))
    #############################################################
    # Build Dataset
    logger.info(f"loading dataset..")
    train_dataset = customDataset(data_file_pathname, idx_trva, label_type, task_name)
    dev_dataset = customDataset(data_file_pathname, idx_te, label_type, task_name)
    # Train the model
    train(args, train_dataset, dev_dataset, model)


def train(args, train_dataset, test_dataset, model):
    random_seed = args.random_seed
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    nb_epoch = args.nb_epoch
    ##############################################
    # Settings for task, model, path, etc
    result_path = os.path.join(args.working_path, 'output', args.data_name, str(args.label_type))
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
    set_seed(random_seed)
    #################################################################
    # dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size,num_workers=4)
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size,num_workers=4)
    #################################################################
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #################################################################
    # train
    logger.info(f"training..")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bot = BaseBot(model, train_dataloader, test_dataloader, optimizer,
                 log_dir=result_log_path, log_level=logging.INFO,
                 checkpoint_dir=model_path, echo=False,
                 device=device, use_tensorboard=False, use_amp=False, seed=123, n_gpus=1)
    bot.train(n_epoch=nb_epoch)
    # save model
    bot.save_model()
    # test
    bot.predict(test_dataloader)

def main():
    args = get_arg()
    DATA_NAME = args.data_name
    model_type = args.model_type
    working_path = args.working_path
    data_file_name = args.data_file_name
    # folds_file_name = args.folds_file_name
    # folds_stat_file_name = args.folds_stat_file_name
    # static_features_path = args.static_features_path
    label_type = args.label_type
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
    # folds_file_pathname = os.path.join(data_path, folds_file_name)
    # folds_stat_file_pathname = os.path.join(data_path, folds_stat_file_name)
    ##############################################
    # Load folds
    # folds_file = np.load(folds_file_pathname)
    # folds_stat_file = np.load(folds_stat_file_pathname)
    ##############################################
    # Set tasks
    # if TASK_NAME == 'icd9':
    #     folds = folds_file['folds_ep_icd9_multi'][0]
    #     folds_stat = folds_stat_file['folds_ep_icd9_multi'][0]
    # elif TASK_NAME == 'mor':
    #     folds = folds_file['folds_ep_mor'][label_type]
    #     folds_stat = folds_stat_file['folds_ep_mor'][label_type]
    # elif TASK_NAME == 'los':
    #     folds = folds_file['folds_ep_mor'][0]
    #     folds_stat = folds_stat_file['folds_ep_mor'][0]
    # tsfstds = []
    # if use_sapsii_scores:
    #     for tr, va, ts in folds[0]:
    #         tsfstds.append(SAPSIITransformer(np.concatenate((tr, va))))
    # else:
    # for serial, non_serial in folds_stat[0]:
    #     tsfstds.append(FoldsStandardizer(serial, non_serial))
    ##############################################
    # Set model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == 1:
        # build model
        # if remove_sapsii:
        #     n_features -= (114-99)
        logger.info(f"model type: HMM, without_static: {without_static}, time_step: {time_step}")
        model = HierarchicalMultimodal(static = not without_static, size_Xs= 5, dropout = dropout, batch_normalization = batch_normalization,
                                       time_step = time_step, n_features = n_features, fit_parameters = fit_parameters)
        model.to(device)
        run_folds(args, model)
    elif model_type == 2:
        # X_static = np.genfromtxt(os.path.join(static_features_path), delimiter=',')
        # sftsflist = []
        # for trainidx, valididx, testidx in folds[0]:
        #     X_static_train = X_static[np.concatenate([trainidx, valididx]).astype(np.int).flatten(), :]
        #     tmean = np.nanmean(X_static_train, axis=0)
        #     tstd = np.nanstd(X_static_train, axis=0)
        #     sftsflist.append(StaticFeaturesStandardizer(train_mean=tmean, train_std=tstd))
        # build model
        # if remove_sapsii:
        #     n_features -= (114-99)
        logger.info(f"model type: FFN")
        model = FeedForwardNetwork(n_features=n_features, hidden_dim=static_hidden_dim, ffn_depth=static_ffn_depth, batch_normalization=batch_normalization)
        model.to(device)
        run_folds(args, model)
    ##############################################

if __name__ == "__main__":
    main()

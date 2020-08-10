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

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arg():
    #### Start here for the common configs ####
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('data_name', type=unicode)
    arg_parser.add_argument('task_name', type=unicode)
    arg_parser.add_argument('model_type', type=unicode, default= 1)
    arg_parser.add_argument('data_file_name', type=unicode)
    arg_parser.add_argument('folds_file_name', type=unicode)
    arg_parser.add_argument('folds_stat_file_name', type=unicode)
    arg_parser.add_argument('--static_features_path', type=unicode, default='')
    arg_parser.add_argument('--label_type', type=int, default=0)
    arg_parser.add_argument('--working_path', '-p', type=unicode, default='..')
    # training
    arg_parser.add_argument('--ffn_depth', type=int, default=4)
    arg_parser.add_argument('--merge_depth', type=int, default=0)
    arg_parser.add_argument('--output_dim', type=int, default=4)
    arg_parser.add_argument('--batch_size', type=int, default=20)
    arg_parser.add_argument('--nb_epoch', type=int, default=50)
    arg_parser.add_argument('--early_stopping', type=str, default='True_BestWeight')
    arg_parser.add_argument('--early_stopping_patience', type=int, default=10)
    arg_parser.add_argument('--batch_normalization', type=str, default='False')
    arg_parser.add_argument('--learning_rate', type=float, default=0.001)
    arg_parser.add_argument('--dropout', type=float, default=0.1)
    arg_parser.add_argument('--without_static', action='store_true')
    arg_parser.add_argument('--remove_sapsii', action='store_true')
    arg_parser.add_argument('--static_ffn_depth', type=int, default=2)
    arg_parser.add_argument('--static_hidden_dim', type=int, default=None)
    arg_parser.add_argument('--use_sapsii_scores', action='store_true')

    return arg_parser.parse_args()

def run_folds(args, model, splits, tsf):
    DATA_NAME = args.data_name
    task_name = args.task_name
    data_file_name = args.data_file_name
    working_path = args.working_path
    label_type = args.label_type
    model_type = int(args.model_type)
    #############################################################
    data_path = os.path.join(working_path, 'data', DATA_NAME)
    data_file_pathname = os.path.join(data_path, data_file_name)
    #############################################################
    n_reps = min(len(splits), 5)
    for rep in range(n_reps):
        fold_idxs = splits[rep]
        # n folds in each repeat
        for i_fold, fold_idx in enumerate(fold_idxs):
            # train/validation/test set or train/test set
            if len(fold_idx) == 2:
                idx_trva, idx_te = fold_idx
            elif len(fold_idx) == 3:
                idx_tr, idx_va, idx_te = fold_idx
                idx_trva = np.concatenate((idx_tr, idx_va))
            # Build Dataset
            train_dataset = customDataset(data_file_pathname, idx_trva, tsf, label_type, task_name, model_type)
            dev_dataset = customDataset(data_file_pathname, idx_te, tsf, label_type, task_name, model_type)
            # Train the model
            train(args, train_dataset, dev_dataset, model)


def train(args, train_dataset, dev_dataset, model):
    batch_size = args.batch_size
    nb_epoch = args.nb_epoch
    learning_rate = args.learning_rate
    #################################################################
    random_seed = 12321
    output_dir = os.path.join(args.working_path, 'output', args.data_name, args.data_file_name.split('.')[0], str(args.label_type))
    checkpoint_last = output_dir
    fp16 = False
    n_gpu = 1
    start_step = 0
    start_epoch = 0
    num_train_epochs = nb_epoch
    adam_epsilon = 1e-8
    warmup_steps = 0
    num_training_steps = 10000
    train_batch_size = batch_size
    #################################################################
    set_seed(random_seed)
    #################################################################
    # dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size,num_workers=4)
    # TODO: Build readers, discretizers, normalizers
    #################################################################
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    """
    #################################################################
    # optimizator
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)

    checkpoint_last = os.path.join(output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))
    #################################################################
    """
    if args.local_rank == 0:
        torch.distributed.barrier()
    """
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    #################################################################
    # train
    global_step = start_step
    tr_loss, logging_loss, avg_loss, tr_nb = 0.0, 0.0,0.0,0
    model.zero_grad()
    model.train()

    for idx in range(start_epoch, int(num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            # TODO: model
            inputs,inputs_ids,masks,labels = [x.to(args.device) for x in batch]
            loss = model(inputs,inputs_ids,masks,labels)

            if n_gpu > 1:
                loss = loss.mean()
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step %100 == 0:
                    logger.info("  steps: %s  ppl: %s", global_step, round(avg_loss,5))
                if local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
                    # Log metrics
                    logging_loss = tr_loss
                    tr_nb = global_step

                # validate
                """
                if local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, dev_dataset)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value,4))
                            # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, '{}-{}-{}'.format(checkpoint_prefix, global_step,round(results['perplexity'],4)))

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    #保存模型
                    model_to_save = model.module.encoder if hasattr(model,'module') else model.encoder  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

                    logger.info("Saving linear to %s",os.path.join(args.output_dir, "linear.bin"))
                    model_to_save_linear = model.module.text_linear if hasattr(model, 'module') else model.text_linear
                    torch.save(model_to_save_linear.state_dict(), os.path.join(output_dir, "linear.bin"))
                    logger.info("Saving embeddings to %s",os.path.join(args.output_dir, "embeddings.bin"))
                    model_to_save_embeddings = model.module.text_embeddings if hasattr(model, 'module') else model.text_embeddings
                    torch.save(model_to_save_embeddings.state_dict(), os.path.join(output_dir, "embeddings.bin"))


                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save.save_pretrained(last_output_dir)
                    logger.info("Saving linear to %s",os.path.join(last_output_dir, "linear.bin"))
                    model_to_save_linear = model.module.text_linear if hasattr(model, 'module') else model.text_linear
                    torch.save(model_to_save_linear.state_dict(), os.path.join(last_output_dir, "linear.bin"))
                    logger.info("Saving embeddings to %s",os.path.join(last_output_dir, "embeddings.bin"))
                    model_to_save_embeddings = model.module.text_embeddings if hasattr(model, 'module') else model.text_embeddings
                    torch.save(model_to_save_embeddings.state_dict(), os.path.join(last_output_dir, "embeddings.bin"))
                    logger.info("Saving model to %s",os.path.join(last_output_dir, "model.bin"))
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), os.path.join(last_output_dir, "model.bin"))


                    idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                    with open(idx_file, 'w', encoding='utf-8') as idxf:
                        idxf.write(str( idx) + '\n')
                    torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

                    step_file = os.path.join(last_output_dir, 'step_file.txt')
                    with open(step_file, 'w', encoding='utf-8') as stepf:
                        stepf.write(str(global_step) + '\n')
                """
            if max_steps > 0 and global_step > max_steps:
                break
        if max_steps > 0 and global_step > max_steps:
            break


def main():
    try:
        args = arg_parser.parse_args()
        DATA_NAME = args.data_name
        model_type = int(args.model_type)
        working_path = args.working_path
        data_file_name = args.data_file_name
        folds_file_name = args.folds_file_name
        folds_stat_file_name = args.folds_stat_file_name
        static_features_path = args.static_features_path
        label_type = args.label_type
        fit_parameters = [args.output_dim, args.ffn_depth, args.merge_depth]
        batch_normalization = args.batch_normalization
        dropout = args.dropout
        without_static = args.without_static
        remove_sapsii = args.remove_sapsii
        static_ffn_depth = args.static_ffn_depth
        static_hidden_dim = args.static_hidden_dim
        use_sapsii_scores = args.use_sapsii_scores
        print('DATA_NAME:', DATA_NAME, 'TASK_NAME:', TASK_NAME, 'working_path:', working_path, 'model_type:', model_type)
        ##############################################
        time_step = 48
        n_features = 136
        ##############################################
        # Settings for task, model, path, etc
        data_path = os.path.join(working_path, 'data', DATA_NAME)
        result_path = os.path.join(working_path, 'output', DATA_NAME, data_file_name.split('.')[0], str(label_type))
        result_log_path = os.path.join(result_path, 'log')
        model_path = os.path.join(working_path, 'model', DATA_NAME, data_file_name.split('.')[0], str(label_type))
        for required_path in [result_path, result_log_path, model_path]:
            if not os.path.exists(required_path):
                os.makedirs(required_path)
        folds_file_pathname = os.path.join(data_path, folds_file_name)
        folds_stat_file_pathname = os.path.join(data_path, folds_stat_file_name)
        ##############################################
        # Load folds
        folds_file = np.load(folds_file_pathname)
        folds_stat_file = np.load(folds_stat_file_pathname)
        ##############################################
        # Set tasks
        if TASK_NAME == 'icd9':
            folds = folds_file['folds_ep_icd9_multi'][0]
            folds_stat = folds_stat_file['folds_ep_icd9_multi'][0]
        elif TASK_NAME == 'mor':
            folds = folds_file['folds_ep_mor'][label_type]
            folds_stat = folds_stat_file['folds_ep_mor'][label_type]
        elif TASK_NAME == 'los':
            folds = folds_file['folds_ep_mor'][0]
            folds_stat = folds_stat_file['folds_ep_mor'][0]
        tsfstds = []
        if use_sapsii_scores:
            for tr, va, ts in folds[0]:
                tsfstds.append(SAPSIITransformer(np.concatenate((tr, va))))
        else:
            for serial, non_serial in folds_stat[0]:
                tsfstds.append(FoldsStandardizer(serial, non_serial))
        ##############################################
        # Set model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_type == 1:
            # build model
            if remove_sapsii:
                n_features -= (114-99)
            model = HierarchicalMultimodal(static = not without_static, size_Xs= X_s.shape[1], dropout = dropout, batch_normalization = batch_normalization,
                                           time_step = time_step, n_features = n_features, fit_parameters = fit_parameters)
            model.to(device)
            run_folds(args, model, folds, tsfstds, TASK_NAME)
        elif model_type == 2:
            X_static = np.genfromtxt(os.path.join(static_features_path), delimiter=',')
            sftsflist = []
            for trainidx, valididx, testidx in folds[0]:
                X_static_train = X_static[np.concatenate([trainidx, valididx]).astype(np.int).flatten(), :]
                tmean = np.nanmean(X_static_train, axis=0)
                tstd = np.nanstd(X_static_train, axis=0)
                sftsflist.append(StaticFeaturesStandardizer(train_mean=tmean, train_std=tstd))
            # build model
            if remove_sapsii:
                n_features -= (114-99)
            model = FeedForwardNetwork(n_features=n_features, hidden_dim=static_hidden_dim, ffn_depth=static_ffn_depth, batch_normalization=batch_normalization)
            model.to(device)
            train(args, model, folds, sftsflist)
        ##############################################
    except:
        print("error!")

if __name__ == "__main__":
    main()

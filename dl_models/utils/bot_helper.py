import os, yaml, pickle, argparse
import random, logging, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import cos, pi
from sklearn import metrics

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

class BaseBot():
    ##################################
    def __init__(self, model, train_loader, val_loader, optimizer,
                 log_dir="./cache/logs/", log_level=logging.INFO,
                 checkpoint_dir="./cache/model_cache/", echo=False,
                 device="cuda:0", use_tensorboard=False, use_amp=False, seed=123, n_gpus=1):
        super(BaseBot, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.log_dir = log_dir
        self.log_level = log_level
        self.checkpoint_dir = checkpoint_dir
        self.echo = echo
        self.device = device
        self.use_tensorboard = use_tensorboard
        self.use_amp = use_amp
        self.seed = seed
        self.n_gpus = n_gpus
        self.step = 0
        self.gradient_accumulation_steps = 1
        self.clip_grad = 0
        self.batch_dim = 0
        self.y_task = 2
        ###########################################################
        for path in [self.log_dir, self.checkpoint_dir]:
            if not os.path.exists(path) or not os.path.isdir(path):
                try:
                    os.makedirs(path)
                except:
                    print(f"make {path} failed!")
        ###########################################################
        if self.use_amp:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        if self.n_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        ###########################################################
        # self.logger = Logger(
        #     self.name, str(self.log_dir), self.log_level,
        #     use_tensorboard=self.use_tensorboard, echo=self.echo)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.info("SEED: %s", self.seed)
        ###########################################################
        self.count_model_parameters()
        ###########################################################
        self.set_seed(self.seed)

    ##################################
    def count_model_parameters(self):
        self.logger.info(
            "# of parameters: {:,d}".format(
                np.sum(list(p.numel() for p in self.model.parameters()))))
        self.logger.info(
            "# of trainable parameters: {:,d}".format(
                np.sum(list(p.numel() for p in self.model.parameters() if p.requires_grad))))

    ##################################
    def set_label_type(self, y_task):
        self.y_task = y_task
    ##################################
    def set_loss_function(self, loss_name):
        if loss_name == "CrossEntropyLoss":
            self.logger.info("set loss function :" + loss_name)
            self.criterion = torch.nn.CrossEntropyLoss()
        elif loss_name == "MSELoss":
            self.logger.info("set loss function :" + loss_name)
            self.criterion = torch.nn.MSELoss()
        else:
            self.logger.info("no such loss!")
    ##################################
    def load_model(self, target_path):
        self.model.load_state_dict(torch.load(target_path)["model"])

    ##################################
    def save_model(self):
        paths = os.path.join(self.checkpoint_dir,str(self.step)+".pth")
        states = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step
        }
        torch.save(states, paths)
        self.logger.info("model saved at: %s", paths)

    ##################################
    def get_batch_size(self, batch, batch_dim):
        if isinstance(batch[0], dict):
            for key in batch[0]:
                return batch[0][key].size(batch_dim)
        else:
            return batch[0].size(batch_dim)

    ##################################
    def mse(self, pred, y):
        pred = np.array(pred)
        y = np.array(y)
        mse = metrics.mean_squared_error(y, pred)
        self.logger.info(
            "=" * 20 + "MSE %f" + "=" * 20, mse)
        res = {"mse": mse}
        return res

    ##################################
    def metric_auc(self, pred, y):
        pred = np.array(pred)
        y = np.array(y)
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
        auc_score = metrics.auc(fpr, tpr)
        auprc_score = metrics.average_precision_score(y, pred)
        self.logger.info(
            "=" * 20 + "Auc %f / Prc %f" + "=" * 20, (auc_score, auprc_score))
        res = {"auroc": auc_score, "auprc":auprc_score}
        return res

    ##################################
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if os.environ.get("DETERMINISTIC", None):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    ##################################
    def label_to_device(self, label):
        if self.y_task == 1:
            return label.to(self.device, dtype=torch.float32).squeeze(1)
        else:
            return label.to(self.device, dtype=torch.long).squeeze(1)

    ##################################
    def train_one_step(self, input_tensors, input_tensors2, target):
        self.model.train()
        output = self.model([input_tensors, input_tensors2])
        batch_loss = self.criterion(output, target)
        batch_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # self.logger.info(f"output: {output}, target: {target}.")
        return batch_loss.data.cpu().item(), 0

    ##################################
    def train(self, n_epoch=None):
        ###########################################################
        self.optimizer.zero_grad()
        self.logger.info(
            "Optimizer {}".format(str(self.optimizer)))
        self.logger.info("Batches per epoch: {}".format(
            len(self.train_loader)))
        ###########################################################
        # Train starts
        num_iter = len(self.train_loader)
        totol_step = num_iter * n_epoch
        #while self.step < totol_step:
        with tqdm(range(totol_step)) as pbar:
            for e in range(n_epoch):
                #self.logger.info("=" * 20 + "Epoch %d" + "=" * 20, epoch)
                for i, (input_tensors, input_tensors2, targets) in enumerate(self.train_loader):
                    adjust_learning_rate(e, n_epoch, i, num_iter)
                    input_tensors = input_tensors.to(self.device)
                    input_tensors2 = input_tensors2.to(self.device)
                    targets = self.label_to_device(targets)
                    train_loss, train_weight = self.train_one_step(input_tensors, input_tensors2, targets)
                    # if self.step % 100 == 0:
                    #     self.logger.info("train_loss: %8f" %train_loss)
                    if self.step > totol_step:
                        break
                    self.step += 1
                    pbar.set_description("Loss-%8f" % train_loss)
                    pbar.update(1)
        self.logger.info("finishing training..")

    ##################################
    def train_ffn(self, n_epoch=None):
        ###########################################################
        self.optimizer.zero_grad()
        self.logger.info(
            "Optimizer {}".format(str(self.optimizer)))
        self.logger.info("Batches per epoch: {}".format(
            len(self.train_loader)))
        ###########################################################
        # Train starts
        self.model.train()
        totol_step = len(self.train_loader) * n_epoch
        #while self.step < totol_step:
        with tqdm(range(totol_step)) as pbar:
            for e in range(n_epoch):
                #self.logger.info("=" * 20 + "Epoch %d" + "=" * 20, epoch)
                for input_tensors, targets in self.train_loader:
                    input_tensors = input_tensors.to(self.device)
                    targets = self.label_to_device(targets)
                    output = self.model(input_tensors)
                    batch_loss = self.criterion(output, targets)
                    batch_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # if self.step % 100 == 0:
                    #     self.logger.info("train_loss: %8f" %train_loss)
                    if self.step > totol_step:
                        break
                    self.step += 1
                    train_loss = batch_loss.data.cpu().item()
                    pbar.set_description("Loss-%8f" % train_loss)
                    pbar.update(1)
        self.logger.info("finishing training..")

    ##################################
    def eval(self, loader):
        self.model.eval()
        preds, ys = [], []
        losses, weights = [], []
        self.logger.debug("Evaluating...")
        with torch.no_grad():
            for input_tensors, input_tensors2, y_local in tqdm(loader, ncols=100):
                input_tensors = input_tensors.to(self.device)
                input_tensors2 = input_tensors2.to(self.device)
                y_local = self.label_to_device(y_local)
                output = self.model([input_tensors, input_tensors2])
                batch_loss, weights = self.criterion(output, y_local)
                losses.append(batch_loss.data.cpu().item())
                # Save batch labels and predictions
                preds.append(output.cpu())
                ys.append(y_local.cpu())
        loss = np.average(losses, weights=weights)
        metrics = {"loss": (loss, self.loss_format % loss),
                   "predict_y": preds,
                   "gt_y": ys
                   }

        self.logger.info("Eval results: {}".format(metrics))
        return metrics

    ##################################
    def predict(self, loader, return_y=True):
        self.model.eval()
        outputs0, y_global0 = [], []
        with torch.no_grad():
            for input_tensors, input_tensors2, y_local in tqdm(loader, ncols=100):
                input_tensors = input_tensors.to(self.device)
                input_tensors2 = input_tensors2.to(self.device)
                y_local = self.label_to_device(y_local)
                output = self.model.predict0([input_tensors, input_tensors2])

                [outputs0.append(o) for o in output.cpu().numpy()]
                [y_global0.append(o) for o in y_local.cpu().numpy()]

        ###########################################################
        if self.y_task == 2:
            result = self.metric_auc(outputs0, y_global0)
            np.savez(os.path.join(self.log_dir, "results.npz"), result=result)
        else:
            result = self.mse(outputs0, y_global0)
            np.savez(os.path.join(self.log_dir, "results.npz"), result=result)
        ###########################################################
        if return_y:
            return outputs0, y_global0
        return outputs0


    ##################################
    def predict_ffn(self, loader, return_y=True):
        self.model.eval()
        outputs0, y_global0 = [], []
        with torch.no_grad():
            for input_tensors, y_local in tqdm(loader, ncols=100):
                input_tensors = input_tensors.to(self.device)
                y_local = self.label_to_device(y_local)
                output = self.model.predict0(input_tensors)
                [outputs0.append(o) for o in output.cpu().numpy()]
                [y_global0.append(o) for o in y_local.cpu().numpy()]

        ###########################################################
        if self.y_task == 2:
            result = self.metric_auc(outputs0, y_global0)
            np.savez(os.path.join(self.log_dir, "results.npz"), result=result)
        else:
            result = self.mse(outputs0, y_global0)
            np.savez(os.path.join(self.log_dir, "results.npz"), result=result)
        ###########################################################
        if return_y:
            return outputs0, y_global0
        return outputs0


    def adjust_learning_rate(self, epoch, n_epochs, iteration, num_iter, lr_decay = 'cos', warmup = True):

        # original learning rate
        origin_lr = self.lr
        # learning rate is multiplied by gamma on schedule
        gamma  = 0.1
        # decrease learning rate at these epochs
        schedule = [100, 150, 200]

        warmup_epoch = 5 if warmup else 0
        warmup_iter = warmup_epoch * num_iter
        current_iter = iteration + epoch * num_iter
        max_iter = n_epochs * num_iter

        if lr_decay == 'step':
            lr = origin_lr * (gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
        elif lr_decay == 'cos':
            lr = origin_lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
        elif lr_decay == 'linear':
            lr = origin_lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
        elif lr_decay == 'schedule':
            count = sum([1 for s in schedule if s <= epoch])
            lr = origin_lr * pow(gamma, count)
        else:
            raise ValueError('Unknown lr mode {}'.format(lr_decay))

        if epoch < warmup_epoch:
            lr = origin_lr * current_iter / warmup_iter

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


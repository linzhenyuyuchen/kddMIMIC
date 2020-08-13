import os, yaml, pickle, argparse
import random, logging, json
import numpy as np
import pandas as pd
from tqdm import tqdm
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
        ###########################################################

    ##################################
    def count_model_parameters(self):
        self.logger.info(
            "# of parameters: {:,d}".format(
                np.sum(list(p.numel() for p in self.model.parameters()))))
        self.logger.info(
            "# of trainable parameters: {:,d}".format(
                np.sum(list(p.numel() for p in self.model.parameters() if p.requires_grad))))

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
    def metric_auc(self, pred, y):
        pred = np.array(pred)
        y = np.array(y)
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
        auc_score = metrics.auc(fpr, tpr)
        self.logger.info(
            "=" * 20 + "Auc %f" + "=" * 20, auc_score)
        return auc_score
        # f1 = metrics.f1_score(y, pred)
        # acc = metrics.accuracy_score(y, y_pred)
        # prec = metrics.precision_score(y, y_pred)
        # rec = metrics.recall_score(y, y_pred)
        # result = {
        #     "accuracy: ": acc,
        #     "precision: ": prec,
        #     "recall: ": rec,
        #     "f1_score: ": f1,
        #     "auc: ": auc_score
        # }
        # self.logger.info("Result: ", result)

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
        return label.to(self.device, dtype=torch.long).squeeze(1)

    ##################################
    def train_one_step_clip(self, input_tensors, input_tensors2, target):
        self.model.train()
        output = self.model([input_tensors, input_tensors2])
        batch_loss = self.criterion(output, target) / self.gradient_accumulation_steps
        if self.use_amp:
            with amp.scale_loss(
                batch_loss, self.optimizer,
                delay_unscale=self.step % self.gradient_accumulation_steps != 0
            ) as scaled_loss:
                scaled_loss.backward()
        else:
            batch_loss.backward()
        if self.step % self.gradient_accumulation_steps == 0:
            if self.clip_grad > 0:
                if not self.use_amp:
                    for param_group in self.optimizer.param_groups:
                        clip_grad_norm_(param_group["params"], self.clip_grad)
                else:
                    clip_grad_norm_(amp.master_params(
                        self.optimizer), self.clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return batch_loss.data.cpu().item() * self.gradient_accumulation_steps,\
               self.get_batch_size(input_tensors, self.batch_dim)

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
        totol_step = len(self.train_loader) * n_epoch
        epoch = 0
        while self.step < totol_step:
            epoch += 1
            self.logger.info(
                "=" * 20 + "Epoch %d" + "=" * 20, epoch)
            for input_tensors, input_tensors2, targets in self.train_loader:
                input_tensors = input_tensors.to(self.device)
                input_tensors2 = input_tensors2.to(self.device)
                targets = self.label_to_device(targets)
                train_loss, train_weight = self.train_one_step(input_tensors, input_tensors2, targets)
                if self.step % 100 == 0:
                    self.logger.info("train_loss: %8f" %train_loss)
                if self.step > totol_step:
                    break
                self.step += 1
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
    def predict(self, loader, return_y=False):
        self.model.eval()
        outputs1, y_global1 = [], []
        outputs0, y_global0 = [], []
        with torch.no_grad():
            for input_tensors, input_tensors2, y_local in tqdm(loader, ncols=100):
                input_tensors = input_tensors.to(self.device)
                input_tensors2 = input_tensors2.to(self.device)
                y_local = self.label_to_device(y_local)
                output = self.model.predict0([input_tensors, input_tensors2])

                [outputs0.append(o) for o in output.cpu().numpy()]
                [y_global0.append(o) for o in y_local.cpu().numpy()]

                outputs1.append(output.cpu())
                y_global1.append(y_local.cpu())
        ###########################################################
        result = self.metric_auc(outputs0, y_global0)
        np.savez(os.path.join(self.log_dir, "results.npz"), result=result)
        ###########################################################
        if return_y:
            return outputs1, y_global1
        return outputs1



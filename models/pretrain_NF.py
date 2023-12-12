"""
Train\Test helper, based on awesome previous work by https://github.com/amirmk89/gepc
"""

import os
import time
import shutil
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import time
import numpy as np
from utils.train_utils import dump_args


def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return new_lr


def compute_loss(nll, reduction="mean", mean=0):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "logsumexp":
        losses = {"nll": torch.logsumexp(nll, dim=0)}
    elif reduction == "exp":
        losses = {"nll": torch.exp(torch.mean(nll) - mean)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses


class Trainer:
    def __init__(self, args, model, train_loader, test_loader,
                 optimizer_f=None, scheduler_f=None):
        self.model = model
        #self.img_model = img_model
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        # Loss, Optimizer and Scheduler
        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)
        #self.img_optimizer =  torch.optim.SGD(self.img_model.parameters(), lr=0.001, momentum=0)
    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            if self.args.lr:
                return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        elif self.args.optimizer == 'adamx':
            if self.args.lr:
                return optim.Adamax(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adamax(self.model.parameters())
        return optim.SGD(self.model.parameters(), lr=self.args.lr)

    def adjust_lr(self, epoch):
        return adjust_lr(self.optimizer, epoch, self.args.model_lr, self.args.model_lr_decay, self.scheduler)

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        """
        state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        state['args'] = self.args

        path_join = os.path.join(self.args.ckpt_dir, filename)
        torch.save(state, path_join)
        if is_best:
            shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_beat.pth.tar'))

    def load_checkpoint(self, filename):
        filename = filename
        try:
            checkpoint = torch.load(filename)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model.set_actnorm_init()
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(filename, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))

    def train(self, log_writer=None, clip=100):
        time_str = time.strftime("%b%d_%H%M_")
        print("_________________",time_str)
        checkpoint_filename = time_str + '_checkpoint.pth.tar'
        dump_args(self.args, self.args.ckpt_dir,time_str)
        start_epoch = 0
        num_epochs = self.args.epochs
        self.model.train()
        self.model = self.model.to(self.args.device)
        #self.img_model.train()
        #self.img_model = self.img_model.to(self.args.device)
        key_break = False
        for epoch in range(start_epoch, num_epochs):
            if key_break:
                break
            print("Starting Epoch {} / {}".format(epoch + 1, num_epochs))
            pbar = tqdm(self.train_loader)
            np_save = None
            for itern, data_arr in enumerate(pbar): ###

                try:
                    data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
                    score = data[-2].amin(dim=-1)
                    label = data[-1]  ### 有监督的label全为1
                    #print(len(data),len(data[0]),len(data[0][0]),len(data[0][0][0]),len(data[0][0][0][0])) ## 最多维度 4 256 3 24  18
                    if self.args.model_confidence:
                        samp = data[0]
                    else:
                        samp = data[0][:, :2]
                    ### 姿态估计
                    self.optimizer.zero_grad()
                    z, nll = self.model(samp.float(), label=label, score=score)  ###需要将unet放在这里面，使得图片数据与姿态重合
                    if nll is None:
                        continue
                    if self.args.model_confidence: ## 默认不启动，可以启动跑一下结果试试
                        nll = nll * score
                    losses = compute_loss(nll, reduction="mean")["total_loss"]
                    losses.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.optimizer.step()

                    pbar.set_description("Loss: {}".format(losses.item()))
                    log_writer.add_scalar('NLL Loss', losses.item(), epoch * len(self.train_loader) + itern)

                except KeyboardInterrupt:
                    print('Keyboard Interrupted. Save results? [yes/no]')
                    choice = input().lower()
                    if choice == "yes":
                        key_break = True
                        break
                    else:
                        exit(1)

            self.save_checkpoint(epoch, filename=checkpoint_filename)
            new_lr = self.adjust_lr(epoch)
            print('Checkpoint Saved. New LR: {0:.3e}'.format(new_lr))

    def test(self):
        self.model.eval()
        self.model.to(self.args.device)
        #self.img_model.eval()
        #self.img_model.to(self.args.device)
        pbar = tqdm(self.test_loader)
        probs = torch.empty(0).to(self.args.device)
        print("Starting Test Eval？？")

        for itern, data_arr in enumerate(pbar):
            data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
            score = data[-2].amin(dim=-1)  ###  这句话表示什么意思? 是标签吗？
            if self.args.model_confidence:
                samp = data[0]
            else:
                samp = data[0][:, :2]

            with torch.no_grad():
                z, nll = self.model(samp.float(), label=torch.ones(data[0].shape[0]), score=score)
            if self.args.model_confidence:
                nll = nll * score
            probs = torch.cat((probs, -1 * nll), dim=0) ### 每一次的长度都不一样
        prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
        print("prob_mat_np[0]",prob_mat_np[0])#(16279,) 所有人物在所有视频里的长度
        return prob_mat_np

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        return checkpoint_state

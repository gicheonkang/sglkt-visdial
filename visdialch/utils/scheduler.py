import numpy as np
import torch
import torch.optim as Optim

class cyclic_lr():
    """
    code for cyclir learning rate
    """
    def __init__(self, iter_per_epoch, base_lr, max_lr, epochs_per_cycle = 2):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.epochs_per_cycle = epochs_per_cycle
        self.iterations_per_epoch = iter_per_epoch
        self.step_size = (self.epochs_per_cycle*self.iterations_per_epoch)/2

    def iteration(self, epoch, batch_idx):
        return epoch*self.iterations_per_epoch + batch_idx

    def lr(self, epoch, batch_idx):
        cycle = np.floor(1+self.iteration(epoch, batch_idx)/(2*self.step_size))
        x = np.abs(self.iteration(epoch, batch_idx)/self.step_size - 2*cycle + 1)
        lr = self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))
        return lr

# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
class WarmupOptimizer(object):
    def __init__(self, lr_base, optimizer, data_size, batch_size):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size


    def step(self):
        self._step += 1

        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

        self.optimizer.step()


    def zero_grad(self):
        self.optimizer.zero_grad()


    def rate(self, step=None):
        if step is None:
            step = self._step

        if step <= int(self.data_size / self.batch_size * 1):
            r = self.lr_base * 1/4.
        elif step <= int(self.data_size / self.batch_size * 2):
            r = self.lr_base * 2/4.
        elif step <= int(self.data_size / self.batch_size * 3):
            r = self.lr_base * 3/4.
        else:
            r = self.lr_base

        return r


def get_optim(config, model, data_size, lr_base=None):
    if lr_base is None:
        lr_base = config["solver"]["initial_lr"]

    return WarmupOptimizer(
        lr_base,
        Optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0,
            betas=(0.9, 0.98),
            eps=1e-9
        ),
        data_size,
        config["solver"]["batch_size"]
    )


def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r

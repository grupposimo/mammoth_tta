# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
from utils.tta.tta_losses import EntropyLoss


class Tent(ContinualModel):
    NAME = 'tent'
    COMPATIBILITY = ['continual-tta']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Fully Test-Time Adaptation.')
        parser.add_argument('--episodic', type=int, default=0, choices=(0, 1),
                            help='Wheter to perform continual or episodic adaptation.')
        parser.add_argument('--steps', type=int, default=1,
                            help='The number of steps to perform during adaptation.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

        if self.args.loadcheck:
            assert Path(self.args.loadcheck).resolve().exists(), f"Checkpoint {self.args.loadcheck} not found"
            self.load_state_dict(torch.load(Path(self.args.loadcheck).resolve()))

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        if self.args.episodic:
            self.reset()
        acc = 0
        for _ in range(self.args.steps):
            self.opt.zero_grad()

            outputs = self.net(inputs)

            loss = self.loss(outputs)
            loss.backward()
            self.opt.step()
            acc += loss.item()

        tot_loss = acc / self.args.steps

        return tot_loss

    def reset(self):
        load_model_and_optimizer(self.net, self.opt, self.model_state, self.optimizer_state)

    def configure_model(self):
        """
        Tent simply functions in 2 steps: prepare the model by disabling gradient to all module but the batch norm.
        Then collect affine transformation parameters to update.
        """
        # Train mode
        self.net.train()
        # Disable all parameters gradiants
        self.net.requires_grad_(False)

        # Enable gradient only for the nn.BatchNorm2d layers
        for module in self.net.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.requires_grad_(True)
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

        # Tents only collects affine transformation parameters
        params = []
        for _, module in self.net.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                for name, parameter in module.named_parameters():
                    if name in ['weight', 'bias']:
                        params.append(parameter)

        # Configure the optimizer
        if self.args.optimizer == 'adam':
            self.opt = optim.Adam(params, lr=self.args.lr, betas=(self.args.betas, 0.999), weight_decay=self.args.optim_wd)
        elif self.args.optimizer == 'sgd':
            self.opt = optim.SGD(params, lr=self.args.lr, momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)
        elif self.args.optimizer == 'adamw':
            self.opt = optim.AdamW(params, lr=self.args.lr, betas=(self.args.betas, 0.999), weight_decay=self.args.optim_wd)

        self.loss = EntropyLoss()

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.net, self.opt)


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

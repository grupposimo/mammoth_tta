# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


class Tent(ContinualModel):
    NAME = 'tent'
    COMPATIBILITY = ['continual-tta']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Fully Test-Time Adaptation.')
        parser.add_argument('--episodic', type=int, required=True,
                            help='Wheter to perform continual or episodic adaptation.')
        parser.add_argument('--optimizer', type=str, required=True, choices=['adam', 'sgd'],
                            help='The optimizer to use.')
        parser.add_argument('--steps', type=float, required=True,
                            help='The number of steps to perform during adaptation.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

        self.configure_model()
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.net, self.opt)

    def observe(self, inputs, labels):

        if self.args.episodic:
            self.reset()

        for _ in range(self.args.steps):
            self.opt.zero_grad()

            outputs = self.net(inputs)

            loss = softmax_entropy(outputs).mean(0)
            loss.backward()
            self.opt.step()
            tot_loss = loss.item()

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
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(True)
                module.track_running_stats(False)
                module.running_mean = None
                module.running_var = None

        # Tents only collects affine transformation parameters
        params = []
        for _, module in self.net.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                for name, parameter in module.named_parameters():
                    if name in ['weight', 'bias']:
                        params.append(parameter)

        # Configure the optimizer
        if self.args.optimizer == 'adam':
            self.opt = optim.Adam(params, lr=self.args.lr, betas=(self.args.betas, 0.999), weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'sgd':
            self.opt = optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits.""" 
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
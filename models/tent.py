# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn as nn
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser

class Tent(ContinualModel):
    NAME = 'tent'
    COMPATIBILITY = ['continual-tta']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Fully Test-Time Adaptation.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()

        outputs = self.net(inputs)

        loss = self.loss(outputs, labels)
        loss.backward()
        tot_loss = loss.item()

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

            buf_outputs = self.net(buf_inputs)
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward()
            tot_loss += loss_mse.item()

            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

            buf_outputs = self.net(buf_inputs)
            loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
            loss_ce.backward()
            tot_loss += loss_ce.item()

        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return tot_loss

    def configure_model(model):
        '''
        Tent simply functions in 2 steps: prepare the model by disabling gradient to all module but the batch norm.
        Then collect affine transformation parameters to update.
        '''
        # Train mode
        model.train()
        # Disable all parameters gradiants
        model.requires_grad_(False)

        # Enable gradient only for the nn.BatchNorm2d layers
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(True)
                module.track_running_stats(False)
                module.running_mean = None
                module.running_var = None

        # Tents only collects affine transformation parameters
        params = []
        for _, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                for name, parameter in module.named_parameters():
                    if name in ['weight', 'bias']:
                        params.append(parameter)

        return model, params
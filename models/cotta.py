# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy

import PIL
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import utils.tta.tta_transforms as my_transforms
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
from utils.tta.tta_losses import EmaEntropyLoss


class Cotta(ContinualModel):
    NAME = 'cotta'
    COMPATIBILITY = ['continual-tta']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual Test-Time Adaptation')
        parser.add_argument('--episodic', type=int, default=0, choices=(0, 1),
                            help='Wheter to perform continual or episodic adaptation.')
        parser.add_argument('--steps', type=int, default=1,
                            help='The number of steps to perform during adaptation.')
        parser.add_argument('--mt-alpha', type=float, default='0.99',
                            help='The alpha parameter for the moving average.')
        parser.add_argument('--rst-m', type=float, default='0.1',
                            help='The momentum for the running mean.')
        parser.add_argument('--ap', type=float, default='0.9',
                            help='The alpha parameter for the affine parameters.')
        parser.add_argument('--augmentation-number', type=int, default=32,
                            help='The number of augmentations to use.')
        parser.add_argument('--restore', type=bool, default=True,
                            help='Whether to restore the source model.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        if self.args.episodic:
            self.reset()
        acc = 0
        for _ in range(self.args.steps):
            outputs = self.net(inputs)

            anchor_prob = nn.functional.softmax(self.model_anchor(inputs), dim=1).max(1)[0]
            standard_ema = self.ema_model(inputs)

            ema_ouptuts = []
            for _ in range(self.args.augmentation_number):
                outputs_ = self.ema_model(self.transforms(inputs)).detach()
                ema_ouptuts.append(outputs_)

            # Threshold choice discussed in supplementary
            if anchor_prob.mean(0) < self.args.ap:
                outputs_ema = torch.stack(ema_ouptuts).mean(0)
            else:
                outputs_ema = standard_ema

            # Student update
            loss = self.loss(outputs, outputs_ema)
            loss.backward()
            acc += loss.item()
            self.opt.step()
            self.opt.zero_grad()

            # Teacher update
            self.update_ema_model()

            # CoTTA stochastic restore of source model information
            if self.args.restore:
                for nm, m in self.net.named_modules():
                    for npp, p in m.named_parameters():
                        if npp in ['weight', 'bias'] and p.requires_grad:
                            mask = (torch.rand(p.shape) < self.args.rst_m).float().cuda()
                            with torch.no_grad():
                                p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1. - mask)

        tot_loss = acc / self.args.steps

        return tot_loss

    def reset(self):
        load_model_and_optimizer(self.net, self.opt, self.model_state, self.optimizer_state)
        # Reset also the teacher model
        self.model_state, self.optimizer_state, self.ema_model, self.model_anchor = copy_model_and_optimizer(self.net, self.opt)

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
                module.track_running_stats(False) if isinstance(module, nn.BatchNorm2d) else None
                module.running_mean = None
                module.running_var = None
            else:
                module.requires_grad_(True)

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

        self.loss = EmaEntropyLoss()

        self.model_state, self.optimizer_state, self.ema_model, self.model_anchor = copy_model_and_optimizer(self.net, self.opt)
        self.transforms = get_tta_transforms()

    def update_ema_model(self):
        for ema_param, param in zip(self.ema_model.parameters(), self.net.parameters()):
            ema_param.data[:] = self.args.mt_alpha * ema_param[:].data[:] + (1 - self.args.mt_alpha) * param[:].data[:]


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def get_tta_transforms(gaussian_std: float = 0.005, soft=False, clip_inputs=False, use_vit=True):
    img_shape = (32, 32, 3) if not use_vit else (224, 224, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0),
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1 / 16, 1 / 16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms

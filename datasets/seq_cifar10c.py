# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import download_and_extract_archive

from backbone.ResNet18 import resnet18
from datasets.seq_cifar10 import MyCIFAR10
from datasets.seq_tinyimagenet import base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_tta_loaders)
from torchvision.transforms.functional import InterpolationMode
from timm.models.vision_transformer import vit_base_patch16_224

CORRUPTIONS = ('gaussian_noise', 'shot_noise', 'impulse_noise',
               'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
               'snow', 'frost', 'fog', 'brightness', 'contrast',
               'elastic_transform', 'pixelate', 'jpeg_compression')


class CIFAR10C(Dataset):
    """CIFAR10-C Dataset.

    Args:
        root (str): root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        corruption (str): corruption to apply to the dataset.
        severity (int): severity of the corruption.
        transform (torchvision.transforms): transformations to apply to the dataset.
        # https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1
    """
    filename = "CIFAR-10-C.tar"
    url = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
    tgz_md5 = "56bf5dcef84df0e2308c6dcbcbbd8499"

    def __init__(self, root, corruption, transform=None, target_transform=None,
                 download=True, severity=5) -> None:
        self.corruption = corruption
        self.severity = severity
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        assert 1 <= severity <= 5
        n_total_cifar = 10000

        if download:
            if self._check_integrity():
                print("Files already downloaded and verified")
            else:
                download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5, remove_finished=True)

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        labels_path = Path(self.root, 'labels.npy')
        labels = np.load(labels_path)

        corruption_file_path = Path(self.root, corruption + '.npy')

        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_cifar:severity * n_total_cifar]
        self.data = images[:n_total_cifar]
        self.targets = labels[:n_total_cifar]

    def __len__(self) -> int:
        return len(self.targets)

    def _check_integrity(self) -> bool:
        if any([not Path(self.root, f + '.npy').exists() for f in CORRUPTIONS + ('labels',)]):
            return False
        return True

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR10C(ContinualDataset):
    """Sequential CIFAR10 Dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformations to apply to the dataset.
    """

    NAME = 'seq-cifar10c'
    SETTING = 'continual-tta'
    N_CLASSES_PER_TASK = 10
    N_TASKS = len(CORRUPTIONS)
    N_CLASSES = 10
    SIZE = (32, 32)
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])
    TEST_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Class method that returns the train and test loaders."""
        transform = self.TRANSFORM

        train_dataset = CIFAR10C(base_path() + 'CIFAR10C', corruption=CORRUPTIONS[self.c_task + 1],
                                 download=True, transform=transform)
        print(CORRUPTIONS[self.c_task + 1])
        train = store_tta_loaders(train_dataset, self)
        return train, None

    def get_source_dataset(self):

        return CIFAR10(base_path() + 'CIFAR10', train=True, transform=self.TRANSFORM), CIFAR10(base_path() + 'CIFAR10', train=False, transform=self.TEST_TRANSFORM)

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR10C.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR10C.N_CLASSES_PER_TASK
                        * SequentialCIFAR10C.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_source_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCIFAR10C.MEAN, SequentialCIFAR10C.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCIFAR10C.MEAN, SequentialCIFAR10C.STD)
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 64


class SequentialCIFAR10C224(SequentialCIFAR10C):
    """Sequential CIFAR10 Dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformations to apply to the dataset.
    """

    NAME = 'seq-cifar10c-224'
    SETTING = 'continual-tta'
    SIZE = (224, 224)
    N_CLASSES_PER_TASK = 10
    N_TASKS = len(CORRUPTIONS)
    N_CLASSES = 10
    MEAN, STD = (0, 0, 0), (1, 1, 1)
    TRANSFORM = transforms.Compose(
        [transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])
    TEST_TRANSFORM = transforms.Compose(
        [transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])

    @staticmethod
    def get_backbone():
        return vit_base_patch16_224(pretrained=True, num_classes=10)


if __name__ == '__main__':
    dataset = CIFAR10C(root=base_path(), download=True, corruption='gaussian_noise', severity=1)

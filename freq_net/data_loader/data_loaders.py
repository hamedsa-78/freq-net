from typing import Tuple
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path
import os
import sys
import shutil
from PIL import Image

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(f"{str(root)}/../")

from freq_net.base import BaseDataLoader
from freq_net.utils import download_url


class DIV2KDataset(Dataset):
    def __init__(
        self,
        root: str = "datasets",
        train=True,
        download=True,
        lr_transform=None,
        hr_transform=None,
    ) -> None:
        self.root = os.path.join(root, "div2k")
        self.train = train
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        if self.train:
            lr_url = (
                "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip"
            )
            hr_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
            lr_output = "DIV2K_train_LR_bicubic_X4.zip"
            hr_output = "DIV2K_train_HR.zip"
            lr_path = "DIV2K_train_LR_bicubic/X4"
            hr_path = "DIV2K_train_HR"
        else:
            lr_url = (
                "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip"
            )
            hr_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
            lr_output = "DIV2K_valid_LR_bicubic_X4.zip"
            hr_output = "DIV2K_valid_HR.zip"
            lr_path = "DIV2K_valid_LR_bicubic/X4"
            hr_path = "DIV2K_valid_HR"

        lr_output = os.path.join(self.root, lr_output)
        hr_output = os.path.join(self.root, hr_output)
        self.lr_path = os.path.join(self.root, lr_path)
        self.hr_path = os.path.join(self.root, hr_path)

        if download:
            if not os.path.exists(lr_output):
                download_url(lr_url, lr_output)
                shutil.unpack_archive(lr_output, self.root)

            if not os.path.exists(hr_output):
                download_url(hr_url, hr_output)
                shutil.unpack_archive(hr_output, self.root)

    def __len__(self):
        return 800 if self.train else 100

    def __getitem__(self, index):
        """Outputs (lr, hr) pairs"""
        lr_img_path = os.path.join(self.lr_path, f"{index + 1:04d}x4.png")
        hr_img_path = os.path.join(self.hr_path, f"{index + 1:04d}.png")

        lr_image = Image.open(lr_img_path)
        hr_image = Image.open(hr_img_path)
        if self.lr_transform:
            lr_image = self.lr_transform(lr_image)
        if self.hr_transform:
            hr_image = self.hr_transform(hr_image)
        return lr_image, hr_image


class DIV2KDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        batch_size,
        data_dir="datasets",
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        train=True,
    ):
        transform = transforms.Compose(
            [
                transforms.Resize(
                    1024,
                    antialias=True,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop((1024, 1024)),
                transforms.Lambda(lambda img: img.convert("YCbCr")),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.44285116, 0.48022078, 0.51065065],
                    [0.22575448, 0.06186319, 0.058383],
                ),
            ]
        )
        self.data_dir = data_dir
        self.dataset = DIV2KDataset(
            self.data_dir,
            train=train,
            download=True,
            lr_transform=transform,
            hr_transform=transform,
        )
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )

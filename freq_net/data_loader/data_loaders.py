from typing import Tuple
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import os
import sys
import shutil

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(f"{str(root)}/../")

from freq_net.base import BaseDataLoader
from freq_net.utils import download_url



class DIV2KDataset(Dataset):
    def __init__(self, root: str='datasets', train=True, download=True, transform=None) -> None:
        self.root = os.path.join(root, 'div2k')
        self.train = train
        self.transform = transform
        
        if self.train:
            lr_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip'
            hr_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'
            subpath = 'train'
        else:
            lr_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip'
            hr_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'
            subpath = 'valid'
            
        path = os.path.join(self.root, subpath)
        lr_output = os.path.join(path, 'lr.zip')
        hr_output = os.path.join(path, 'hr.zip')
        self.lr_path = os.path.join(path, 'lr')
        self.hr_path = os.path.join(path, 'hr')
        
        if download:
            if not os.path.exists(lr_output):
                download_url(lr_url, lr_output)
                shutil.unpack_archive(lr_output, self.lr_path)

            if not os.path.exists(hr_output):
                download_url(hr_url, hr_output)
                shutil.unpack_archive(hr_output, self.hr_path)


    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Outputs (lr, hr) pairs """
        lr_img_path = os.path.join(self.lr_path, f'{index:04d}x4.png')
        hr_img_path = os.path.join(self.hr_path, f'{index:04d}.png')
        lr_image = read_image(lr_img_path)
        hr_image = read_image(hr_img_path)
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        return lr_image, hr_image


class DIV2KDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        batch_size,
        data_dir='datasets',
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        train=True,
    ):
        trsfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.data_dir = data_dir
        self.dataset = DIV2KDataset(
            self.data_dir, train=train, download=True, transform=trsfm
        )
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )

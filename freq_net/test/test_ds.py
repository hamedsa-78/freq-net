import sys
from pathlib import Path
import numpy as np
import torch

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(f"{str(root)}/../")

from freq_net.data_loader.data_loaders import DIV2KDataset, DIV2KDataLoader


ds = DIV2KDataset()
dl = DIV2KDataLoader(16, num_workers=0)
print(next(iter(dl))[0][0].shape)
print(next(iter(dl))[0][1].shape)
print(next(iter(dl))[0][2].shape)


# calculate dataloader normalization
dl = DIV2KDataLoader(1, num_workers=0)

mean, var = np.zeros((3, )), np.zeros((3, ))
for i, img in enumerate(dl):
    X: torch.Tensor = img[0][1]
    mean += X.mean(dim=[0, 2, 3]).numpy()
    var += X.var(dim=[0, 2, 3]).numpy()
    if i % 100 == 99:
        print('image', i + 1)

mean = mean / len(ds)
std = np.sqrt(var / len(ds))
print(f'{mean = }, {std = }')

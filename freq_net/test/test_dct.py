import sys
from pathlib import Path
import numpy as np
import torch

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(f"{str(root)}/../")

from freq_net.model.two_stage_transforms import TwoStageDCT

dcter = TwoStageDCT()
img = torch.randint(0, 255, (3, 1, 512, 512))


img_dct = dcter.dct(img)
print(img_dct.shape)

img_idct = dcter.idct(img_dct)
print(img_idct.shape)

print((img - img_idct).mean().item())
assert (img - img_idct).mean().abs().item() < 1e4

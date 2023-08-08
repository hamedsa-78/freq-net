from pathlib import Path
import sys

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(f"{str(root)}/../")

from freq_net.data_loader.data_loaders import DIV2KDataset, DIV2KDataLoader


ds = DIV2KDataset()
dl = DIV2KDataLoader(2)
print(next(iter(dl)).shape)

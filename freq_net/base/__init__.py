import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(f"{str(root)}/../")

from freq_net.base.base_data_loader import *
from freq_net.base.base_model import *
from freq_net.base.base_trainer import *

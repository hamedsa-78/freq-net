from pathlib import Path
import sys

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(f"{str(root)}/../")

from freq_net.logger.logger import *
from freq_net.logger.visualization import *

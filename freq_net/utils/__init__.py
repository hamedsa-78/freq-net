from pathlib import Path
import sys

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(f"{str(root)}/../")

from freq_net.utils.util import *
from freq_net.utils.download import download_url

from pathlib import Path

from utils import get_datetime_version
from ml_collections import ConfigDict

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent.joinpath('DMLG/g2net/data_full')

config = ConfigDict()
config.checkpoint_path = DATA_ROOT.parent.joinpath('checkpoints/baseline/last.ckpt')
config.seed = 10

config.dataloader = ConfigDict()
config.dataloader.data_path = DATA_ROOT.joinpath("test")
config.dataloader.labels_path = DATA_ROOT.joinpath("test_labels.csv")
config.dataloader.batch_size = 256
config.dataloader.num_workers = 1

config.trainer = ConfigDict()
config.trainer.gpus = 1 #1
config.trainer.auto_select_gpus = True #True

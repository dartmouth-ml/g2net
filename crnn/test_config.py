from pathlib import Path
from ml_collections import ConfigDict

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent.joinpath('DMLG/g2net/data_full')

config = ConfigDict()
config.checkpoint_path = PROJECT_ROOT.joinpath('checkpoints/expand_collate/842021_1438/last.ckpt')
config.submission_path = PROJECT_ROOT.joinpath('submissions')
config.seed = 10

config.dataloader = ConfigDict()
config.dataloader.data_path = DATA_ROOT
config.dataloader.all_labels_path = DATA_ROOT.joinpath("all_labels.csv")
config.dataloader.training_labels_path = DATA_ROOT.joinpath("training_labels.csv")
config.dataloader.validation_labels_path = DATA_ROOT.joinpath("validation_labels.csv")
config.dataloader.test_labels_path = DATA_ROOT.joinpath("test_labels.csv")
config.dataloader.batch_size = 256
config.dataloader.num_workers = 8

config.trainer = ConfigDict()
config.trainer.gpus = 1 #1
config.trainer.auto_select_gpus = True #True

# dummies required for model init
config.model = ConfigDict()
config.model.pretrain = True
config.model.backbone = "resnet18"
config.model.loss_fn = 'CrossEntropy'

config.optimizer = ConfigDict()
config.optimizer.name = "Adam"
config.optimizer.learning_rate = 1e-3

config.scheduler = ConfigDict()
config.scheduler.name = "ReduceLROnPlateau"
config.scheduler.monitor = 'val/loss'
config.scheduler.step_size = 50
config.scheduler.gamma = 0.1
config.scheduler.monitor = 'val/loss'

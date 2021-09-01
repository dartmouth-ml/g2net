from pathlib import Path
from baseline.utils import get_datetime_version
from ml_collections import ConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT.parent.joinpath('DMLG/g2net/data_full')

config = ConfigDict()
config.model_name = "ensemble"
config.version = get_datetime_version()
config.seed = 10

config.dataloader = ConfigDict()
config.dataloader.train_submissions = [
    PROJECT_ROOT.joinpath('ensemble/submissions/baseline_train_sub.csv'),
    PROJECT_ROOT.joinpath('ensemble/submissions/bigmel_train_sub.csv'),
]
config.dataloader.val_submissions = [
    PROJECT_ROOT.joinpath('ensemble/submissions/baseline_val_sub.csv'),
    PROJECT_ROOT.joinpath('ensemble/submissions/bigmel_val_sub.csv'),
]
config.dataloader.batch_size = 64
config.dataloader.num_workers = 8

config.model = ConfigDict()
config.model.n_models = len(config.dataloader.train_submissions)

config.optimizer = ConfigDict()
config.optimizer.name = "Adam"
config.optimizer.learning_rate = 1e-3

config.scheduler = ConfigDict()
config.scheduler.name = None

config.logging = ConfigDict()
config.logging.use_wandb = True
config.logging.project = 'g2net'
config.logging.entity = 'dmlg'
config.logging.tags = [config.version]

config.trainer = ConfigDict()
config.trainer.accelerator = None
config.trainer.gpus = 0
config.trainer.auto_select_gpus = False
config.trainer.min_epochs = 0
config.trainer.max_epochs = 50
config.trainer.val_check_interval = 1000
config.trainer.resume_from_checkpoint = None
config.trainer.fast_dev_run = False
config.trainer.deterministic = False

config.checkpoint = ConfigDict()
config.checkpoint.save_checkpoint = True
config.checkpoint.save_dir = PROJECT_ROOT.joinpath('checkpoints',
                                                   config.model_name,
                                                   config.version)
config.checkpoint.monitor = 'val/AUROC'
config.checkpoint.monitor_mode = 'max'
config.checkpoint.save_last = True
config.checkpoint.save_top_k = 3
config.checkpoint.every_n_steps = 1000

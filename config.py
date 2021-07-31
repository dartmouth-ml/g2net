from pathlib import Path
from utils import get_datetime_version
from ml_collections import ConfigDict

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.joinpath('data')

config = ConfigDict()
config.model_name = "baseline",
config.version = get_datetime_version()
config.seed = 10

config.dataloader = ConfigDict()
config.dataloader.data_path = DATA_ROOT.joinpath("train")
config.dataloader.labels_path = DATA_ROOT.joinpath("training_labels.csv")
config.dataloader.test_data_path = DATA_ROOT.joinpath("test")
config.dataloader.test_labels_path = DATA_ROOT.joinpath("sample_submission.csv")
config.dataloader.val_ratio = 0.2
config.dataloader.batch_size = 64
config.dataloader.num_workers = 0

config.model = ConfigDict()
config.model.pretrain = True
config.model.backbone = "resnet18"

config.policy = ConfigDict()
config.policy.loss_fn = 'BCELoss'
config.policy.optimizer = "Adam"
config.policy.learning_rate = 1e-3
config.policy.scheduler = 'StepLR'
config.policy.scheduler_params = {'step_size': 50, 'gamma': 0.1}

config.logging = ConfigDict()
config.logging.use_wandb = True
config.logging.name = 'baseline'
config.logging.project = 'g2net'
config.logging.entity = 'dmlg'

config.trainer = ConfigDict()
config.trainer.gpus = 1
config.trainer.auto_select_gpus = True
config.trainer.min_epochs = 0
config.trainer.max_epochs = 200,
config.trainer.val_check_interval = 1000,
config.trainer.resume_from_checkpoint = None
config.trainer.fast_dev_run = False
config.trainer.deterministic = False

config.checkpoint = ConfigDict()
config.checkpoint.save_checkpoint = True
config.checkpoint.save_dir = PROJECT_ROOT.joinpath("checkpoints",
                                                   config.model_name,
                                                   config.version)
config.checkpoint.monitor = 'val/loss'
config.checkpoint.monitor_mode = 'min'
config.checkpoint.save_last = True
config.checkpoint.save_top_k = 3
config.checkpoint.every_n_steps = 1000

config.early_stopping = ConfigDict()
config.early_stopping.monitor = 'val/loss'
config.early_stopping.stop_early = True
config.early_stopping.min_delta = 0,
config.early_stopping.patience = 10
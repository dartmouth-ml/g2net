from pathlib import Path
from baseline.utils import get_datetime_version
from ml_collections import ConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT.parent.joinpath('DMLG/g2net/data_full')

config = ConfigDict()
config.model_name = "crnn"
config.version = get_datetime_version()
config.seed = 10

config.dataloader = ConfigDict()
config.dataloader.data_path = DATA_ROOT
config.dataloader.all_labels_path = DATA_ROOT.joinpath("all_labels.csv")
config.dataloader.training_labels_path = DATA_ROOT.joinpath("training_labels.csv")
config.dataloader.validation_labels_path = DATA_ROOT.joinpath("validation_labels.csv")
config.dataloader.test_labels_path = DATA_ROOT.joinpath("test_labels.csv")

config.dataloader.val_ratio = 0.2
config.dataloader.batch_size = 64
config.dataloader.num_workers = 8

config.dataloader.spec_type = 'cqt'
config.dataloader.spec_kwargs = {}

config.dataloader.rescale = None
config.dataloader.bandpass = None
config.dataloader.return_time_series = False

config.model = ConfigDict()
config.model.embedding_dim = 1024
config.model.transformer_nhead = 8
config.model.transformer_dim_feedforward = 1024
config.model.transformer_num_layers = 4
config.model.loss_fn = 'CrossEntropy'

config.optimizer = ConfigDict()
config.optimizer.name = "Adam"
config.optimizer.learning_rate = 1e-3

config.scheduler = ConfigDict()
config.scheduler.name = "CosineAnnealing"
config.scheduler.interval = 'step'

config.logging = ConfigDict()
config.logging.use_wandb = True
config.logging.project = 'g2net'
config.logging.entity = 'dmlg'
config.logging.tags = [config.version]

config.trainer = ConfigDict()
config.trainer.accelerator = None
use_gpu = True
if use_gpu:
    config.trainer.gpus = 1
    config.trainer.auto_select_gpus = True
else:
    config.trainer.gpus = 0
    config.trainer.auto_select_gpus = False
config.trainer.min_epochs = 0
config.trainer.max_epochs = 200
config.trainer.val_check_interval = 1000
config.trainer.resume_from_checkpoint = None
config.trainer.fast_dev_run = False
config.trainer.deterministic = False

config.checkpoint = ConfigDict()
config.checkpoint.save_checkpoint = True
config.checkpoint.save_dir = DATA_ROOT.parent.joinpath('checkpoints',
                                                        config.model_name,
                                                        config.version)
config.checkpoint.monitor = 'val/AUROC'
config.checkpoint.monitor_mode = 'max'
config.checkpoint.save_last = True
config.checkpoint.save_top_k = 3
config.checkpoint.every_n_steps = 1000

config.early_stopping = ConfigDict()
config.early_stopping.early_stop = False
config.early_stopping.monitor = 'val/loss'
config.early_stopping.min_delta = 0
config.early_stopping.patience = 10

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping
)

from ensemble.datamodule import EnsembleDataModule
from ensemble.model import EnsembleModel
from config import config

if config.logging.use_wandb:
    logger = WandbLogger(name=config.model_name,
                         tags=config.logging.tags,
                         project=config.logging.project,
                         entity=config.logging.entity)
else:
    logger = []

callbacks = [LearningRateMonitor()]

if config.checkpoint.save_checkpoint:
    callbacks.append(ModelCheckpoint(dirpath=config.checkpoint.save_dir,
                                     monitor=config.checkpoint.monitor,
                                     mode=config.checkpoint.monitor_mode,
                                     save_last=config.checkpoint.save_last,
                                     save_top_k=config.checkpoint.save_top_k,
                                     every_n_train_steps=config.checkpoint.every_n_steps))

if config.early_stopping.stop_early:
    callbacks.append(EarlyStopping(monitor=config.early_stopping.monitor, 
                                   min_delta=config.early_stopping.min_delta, 
                                   patience=config.early_stopping.patience))

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,

    gpus=config.trainer.gpus,
    auto_select_gpus=config.trainer.auto_select_gpus,
    accelerator=config.trainer.accelerator,

    min_epochs=config.trainer.min_epochs,
    max_epochs=config.trainer.max_epochs,

    val_check_interval=config.trainer.val_check_interval,
    num_sanity_val_steps=0,

    resume_from_checkpoint=config.trainer.resume_from_checkpoint,
    fast_dev_run=config.trainer.fast_dev_run,
    deterministic=config.trainer.deterministic,
)

model = EnsembleModel(config.model, config.optimizer, config.scheduler, config.trainer)
datamodule = EnsembleDataModule(config.dataloader)
datamodule.setup()

trainer.fit(model, datamodule=datamodule)

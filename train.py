import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule import DataModule
from model import LightningG2Net

from config import (
    metadata_config,
    dataloader_config,
    trainer_config,
    model_config,
    policy_config,
    logging_config,
    checkpoint_config,
)

# Training 
if logging_config["use_wandb"]:
    logger = WandbLogger(name=logging_config["name"],
                         tags=logging_config["tags"],
                         project=logging_config["project"],
                         entity=logging_config["entity"])
else:
    logger = []

callbacks = []

if checkpoint_config["save_checkpoint"]:
    callbacks.append(ModelCheckpoint(dirpath=checkpoint_config["save_dir"],
                                     monitor=checkpoint_config["monitor"],
                                     mode=checkpoint_config["mode"],
                                     save_last=checkpoint_config["save_last"],
                                     save_top_k=checkpoint_config["save_top_k"],
                                     every_n_train_steps=checkpoint_config["every_n_train_steps"]))

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,

    gpus=trainer_config["gpus"],
    auto_select_gpus=trainer_config["auto_select_gpus"],

    min_epochs=trainer_config["min_epochs"],
    max_epochs=trainer_config["max_epochs"],
    val_check_interval=trainer_config["val_check_interval"],
    num_sanity_val_steps=0,

    resume_from_checkpoint=trainer_config["resume_from_checkpoint"],
    fast_dev_run=trainer_config["fast_dev_run"],
    deterministic=trainer_config["deterministic"],
    progress_bar_refresh_rate=0
)

datamodule = DataModule(dataloader_config)
datamodule.setup()

model = LightningG2Net(model_config, policy_config)

trainer.fit(model, datamodule=datamodule)

#create_submission(trainer, datamodule, metadata_config)

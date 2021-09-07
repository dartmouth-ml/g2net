from pytorch_lightning import Trainer

from baseline.model import LightningG2Net
from baseline.test_config import config

from common.test import create_submission
from common.dataloaders import StackedSpectogramDM

trainer = Trainer(
    logger=None,
    gpus=config.trainer.gpus,
    auto_select_gpus=config.trainer.auto_select_gpus,
)

model = LightningG2Net.load_from_checkpoint(config.checkpoint_path,
                                            model_config=config.model,
                                            optimizer_config=config.optimizer,
                                            scheduler_config=config.scheduler)
datamodule = StackedSpectogramDM(config.dataloader)

create_submission(
    model,
    trainer,
    datamodule,
    dataloader=None,
    out_dir=config.submission_path
)

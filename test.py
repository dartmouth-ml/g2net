from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from torch.nn.functional import softmax

from model import LightningG2Net
from dataloader import G2NetDataModule
from test_config import config

def create_submission(model, trainer, datamodule):
    submission = pd.DataFrame(columns=["id", "target"])
    model_outs = trainer.predict(model=model,
                                 datamodule=datamodule,
                                 return_predictions=True)

    filenames = model_outs[0]['filename']
    logits = model_outs[0]['logits']
    targets = model_outs[0]['targets']

    ids = [Path(filename).with_suffix('').name for filename in filenames]

    # confidence score for positive class 
    predictions = softmax(logits, dim=-1)[:, 1]

    submission.loc['id'] = ids
    submission.loc['target'] = predictions
    
    submission.to_csv("submission.csv")

if __name__ == "__main__":
    trainer = pl.Trainer(
        logger=None,
        gpus=config.trainer.gpus,
        auto_select_gpus=config.trainer.auto_select_gpus,
    )

    model = LightningG2Net.load_from_checkpoint(config.checkpoint_path,
                                                model_config=config.model,
                                                optimizer_config=config.optimizer,
                                                scheduler_config=config.scheduler)
    datamodule = G2NetDataModule(config.dataloader)

    create_submission(model, trainer, datamodule)

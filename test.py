from pathlib import Path
import pandas as pd
import numpy as np
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
    
    all_ids = []
    all_predictions = []

    for batch in model_outs:
        filenames = batch['filename']
        logits = batch['logits']

        ids = [Path(filename).with_suffix('').name for filename in filenames]

        # confidence score for positive class 
        predictions = softmax(logits, dim=-1)[:, 1]

        all_ids.extend(ids)
        all_predictions.append(predictions.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)

    print(len(all_ids), all_predictions.shape)
    submission['id'] = pd.Series(all_ids)
    submission['target'] = pd.Series(all_predictions)

    print(f'submission shape: {submission.shape}')
    submission.to_csv("submission.csv", index=False, index_label=False)

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

from typing import Union, Optional

from pathlib import Path
import pandas as pd
import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

def create_submission(model: LightningModule,
                      trainer: Trainer,
                      datamodule: Union[LightningDataModule, None],
                      dataloader: Optional[Union[DataLoader, None]],
                      out_dir: Path):
    
    if datamodule is None and dataloader is None:
        raise ValueError('one of datamodule and dataloader cannot be None')

    submission = pd.DataFrame(columns=["id", "target"])

    if datamodule is not None:
        model_outs = trainer.predict(model=model,
                                    datamodule=datamodule,
                                    return_predictions=True)
    elif dataloader is not None:
        model_outs = trainer.predict(model=model,
                                     dataloaders=dataloader,
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
    out_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_dir.joinpath("submission.csv"), index=False, index_label=False)
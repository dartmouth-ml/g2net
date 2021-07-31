# model
import pytorch_lightning as pl
from torchmetrics import (
    MetricCollection,
    Accuracy,
    AUROC,
    F1,
)

import torch
from torch import nn
from torchvision.models import resnet18

from losses import ROCStarLoss

class LightningG2Net(pl.LightningModule):
    def __init__(self, model_config, policy_config):
        super(LightningG2Net, self).__init__()
        self.resnet = resnet18(pretrained=model_config['pretrained']) # remove last layer, fix first layer
        self.output_layer = nn.Linear(1000, 2)

        # hparams
        self.lr = policy_config['lr']
        self.optimizer_name = policy_config['optimizer']
        self.loss_fn_name = policy_config['loss_fn']
        self.loss_fn = self.configure_loss_fn(self.loss_fn_name)

        # metrics
        self.metrics = MetricCollection([
            Accuracy(num_classes=2, threshold=0.5),
            F1(num_classes=2, threshold=0.5),
            AUROC(num_classes=2),
        ])

        # aux metrics that we keep track of
        self.prev_epoch_trues = torch.Tensor()
    
    def configure_loss_fn(self, loss_fn):
        if loss_fn == 'CrossEntropy':
            return nn.BCELoss(weight=None)
        
        elif loss_fn == 'ROC_Star':
            return ROCStarLoss()
        
        else:
            raise NotImplementedError(loss_fn)

    def forward(self, x):
        # resnet
        x = self.resnet(x)
        x = self.output_layer(x)
        
        return x

    def configure_optimizers(self):
        if self.optimizer_name == 'Adam':
            return torch.optim.Adam(self.parameters(), self.lr)

        else:
            raise NotImplementedError(self.optimizer_name)
    
    def on_train_start(self):
        if self.loss_fn_name == 'ROC_Star':
            for batch_idx, batch in enumerate(self.train_dataloader()):
                _, targets = batch
                self.loss_fn.epoch_true_acc[batch_idx] = targets
        
        self.loss_fn.on_epoch_end()

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.forward(inputs)
        preds = torch.argmax(logits, dim=-1)
        loss = self.loss_fn(logits, targets)

        metrics = self.metrics(preds, targets)
        metrics = {f'train/{k}':v for k,v in metrics.items()}

        self.log('train/loss', loss)
        self.log_dict(metrics, on_step=False, on_epoch=True)

        if self.loss_fn_name == 'ROC_Star':
            self.loss_fn.epoch_true_acc[batch_idx] = targets
            self.loss_fn.epoch_pred_acc[batch_idx] = logits
        
        return {'loss': loss, 'logits': logits}
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.forward(inputs)
        preds = torch.argmax(logits, dim=-1)
        loss = self.loss_fn(logits, targets, self.gamma)

        metrics = self.metrics(preds, targets)
        metrics = {f'val/{k}':v for k,v in metrics.items()}

        self.log('val/loss', loss)
        self.log_dict(metrics, on_step=False, on_epoch=True)

        return loss
    
    def on_train_epoch_end(self):
        if self.loss_fn_name == "ROC_Star":
            self.loss_fn.on_epoch_end()
    
    
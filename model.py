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
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    googlenet
)
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR
)
from losses import ROCStarLoss

class LightningG2Net(pl.LightningModule):
    def __init__(self,
                 model_config,
                 policy_config,
                 optimizer_config,
                 scheduler_config):
        super(LightningG2Net, self).__init__()

        self.resnet = self.configure_backbone(model_config.backbone, model_config.pretrain)
        self.output_layer = nn.Linear(1000, 2)

        # hparams
        self.lr = policy_config.lr
        self.optimizer = self.configure_optimizers(optimizer_config, scheduler_config)
        self.loss_fn = self.configure_loss_fn(model_config.loss_fn)

        # metrics
        self.metrics = MetricCollection([
            Accuracy(num_classes=2, threshold=0.5),
            F1(num_classes=2, threshold=0.5),
            AUROC(num_classes=2),
        ])

        # aux metrics that we keep track of
        self.prev_epoch_trues = torch.Tensor()
    
    def configure_backbone(self, backbone, pretrained):
        if backbone == 'resnet18':
            return resnet18(pretrained=pretrained) # remove last layer, fix first layer
        elif backbone == 'resnet34':
            return resnet34(pretrained=pretrained) # remove last layer, fix first layer
        elif backbone == 'resnet50':
            return resnet50(pretrained=pretrained) # remove last layer, fix first layer     
        elif backbone == 'resnet101':
            return resnet101(pretrained=pretrained)
        elif backbone == 'googlenet':
            return googlenet(pretrained=pretrained)
        else:
            raise NotImplementedError(backbone)

    def configure_loss_fn(self, loss_fn):
        if loss_fn == 'CrossEntropy':
            return nn.BCELoss(weight=None)
        
        elif loss_fn == 'ROC_Star':
            return ROCStarLoss()
        
        else:
            raise NotImplementedError(loss_fn)

    def configure_lr_schedulers(self, optimizer, scheduler_config):
        if scheduler_config is None:
            return None
        
        if scheduler_config.name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer)
        
        elif scheduler_config.name == 'StepLR':
            scheduler = StepLR(optimizer, 
                               scheduler_config.step_size,
                               scheduler_config.gamma)

        elif scheduler_config is not None:
            raise NotImplementedError(scheduler_config.name)
        
        return {'scheduler': scheduler, 'monitor': scheduler_config.monitor}

    def configure_optimizers(self, optimizer_config, scheduler_config):
        optimizer_name = optimizer_config.name

        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), self.lr)
        else:
            raise NotImplementedError(optimizer_name)
        
        scheduler_dict = self.configure_lr_schedulers(scheduler_config)

        if scheduler_dict is None:
            return optimizer
        else:
            return {"optimizer": optimizer, 
                    "lr_scheduler": scheduler_dict}
    
    def forward(self, x):
        # resnet
        x = self.resnet(x)
        x = self.output_layer(x)
        
        return x
    
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
    
    
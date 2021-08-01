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
import torch.nn.functional as F
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    googlenet
)
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingLR,
)
from losses import ROCStarLoss

class LightningG2Net(pl.LightningModule):
    def __init__(self,
                 model_config,
                 optimizer_config,
                 scheduler_config):
        super(LightningG2Net, self).__init__()

        self.resnet = self.configure_backbone(model_config.backbone, model_config.pretrain)
        self.resnet.fc = nn.Linear(512, 2)

        # hparams
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.model_config = model_config
        self.loss_fn = self.configure_loss_fn()

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

    def configure_loss_fn(self):
        if self.model_config.loss_fn == 'CrossEntropy':
            return nn.CrossEntropyLoss(weight=None)
        
        elif self.model_config.loss_fn == 'ROC_Star':
            return ROCStarLoss()
        
        else:
            raise NotImplementedError(self.model_config.loss_fn )

    def configure_lr_schedulers(self, optimizer, scheduler_config):
        if scheduler_config is None:
            return None
        
        if scheduler_config.name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer)
        
        elif scheduler_config.name == 'StepLR':
            scheduler = StepLR(optimizer, 
                               scheduler_config.step_size,
                               scheduler_config.gamma)
        
        elif scheduler_config.name == 'CosineAnnealing':
            scheduler = CosineAnnealingLR(optimizer, T_max=len(self.train_dataloader()))

        elif scheduler_config is not None:
            raise NotImplementedError(scheduler_config.name)
        
        return {'scheduler': scheduler, 'monitor': scheduler_config.monitor}

    def configure_optimizers(self):
        if self.optimizer_config.name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), self.optimizer_config.learning_rate)
        else:
            raise NotImplementedError(self.optimizer_config.name)
        
        scheduler_dict = self.configure_lr_schedulers(optimizer, self.scheduler_config)

        if scheduler_dict is None:
            return optimizer
        else:
            return {"optimizer": optimizer, 
                    "lr_scheduler": scheduler_dict}

    def forward(self, x):
        # resnet
        x = self.resnet(x)
        return x
    
    def on_train_start(self):
        if self.model_config.loss_fn == 'ROC_Star':
            for batch_idx, batch in enumerate(self.train_dataloader()):
                _, targets = batch
                self.loss_fn.epoch_true_acc[batch_idx] = targets
        
            self.loss_fn.on_epoch_end()

    def training_step(self, batch, batch_idx):
        inputs, targets, filename = batch
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, targets)

        metrics = self.metrics(logits, targets)
        metrics = {f'train/{k}':v for k,v in metrics.items()}

        self.log('train/loss', loss)
        self.log_dict(metrics, on_step=False, on_epoch=True)

        if self.model_config.loss_fn == 'ROC_Star':
            self.loss_fn.epoch_true_acc[batch_idx] = targets
            self.loss_fn.epoch_pred_acc[batch_idx] = logits
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets, filename = batch
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, targets)

        metrics = self.metrics(logits, targets)
        metrics = {f'val/{k}':v for k,v in metrics.items()}

        self.log('val/loss', loss)
        self.log_dict(metrics, on_step=False, on_epoch=True)

        if self.model_config.loss_fn == 'ROC_Star':
            self.loss_fn.epoch_true_acc[batch_idx] = targets
            self.loss_fn.epoch_pred_acc[batch_idx] = logits

        return loss
    
    def on_train_epoch_end(self):
        if self.model_config.loss_fn == "ROC_Star":
            self.loss_fn.on_epoch_end()
    
    
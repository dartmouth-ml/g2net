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
from torch.nn import functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101, googlenet

from losses import roc_star_loss

class LightningG2Net(pl.LightningModule):
    def __init__(self, model_config, policy_config):
        super(LightningG2Net, self).__init__()

        self.resnet = self.configure_backbone(model_config['backbone'], model_config['resnet_pretrain'])
        self.output_layer = nn.Linear(1000, 2)

        # hparams
        self.lr = policy_config['lr']
        self.optimizer = self.configure_optimizer(policy_config['optimizer'])
        self.loss_fn = self.configure_loss_fn(policy_config['loss_fn'])
        self.scheduler = self.configure_scheduler(policy_config['step_size'], policy_config['gamma'])

        # metrics
        self.metrics = MetricCollection([
            Accuracy(num_classes=2, threshold=0.5),
            F1(num_classes=2, threshold=0.5),
            AUROC(num_classes=2),
        ])
    
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
            return F.binary_cross_entropy_with_logits
        
        elif loss_fn == 'ROC_Star':
            return roc_star_loss
        
        else:
            raise NotImplementedError(loss_fn)

    def configure_scheduler(self, step_size, gamma):
        if self.scheduler_name == 'steplr':
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size,gamma)
        else:
            raise NotImplementedError(self.scheduler_name)

    def configure_optimizers(self, optimizer_name):
        if optimizer_name == 'Adam':
            return torch.optim.Adam(self.parameters(), self.lr)
        else:
            raise NotImplementedError(optimizer_name)
    
    def forward(self, x):
        # resnet
        x = self.resnet(x)
        x = self.output_layer(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.forward(inputs)
        preds = torch.argmax(logits, dim=-1)
        loss = self.loss_fn(logits, targets)

        metrics = self.metrics(preds, targets)
        self.log_dict({
            'train/loss': loss,
             **metrics,
            })
        return {'loss': loss, 'logits': logits}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(y,logits,self.gamma)
        self.log('val_loss', loss)
        return loss
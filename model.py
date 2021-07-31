# model
import pytorch_lightning as pl

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18

from losses import roc_star_loss

class LightningG2Net(pl.LightningModule):
    def __init__(self, model_config, policy_config):
        super(LightningG2Net, self).__init__()
        self.resnet = resnet18(pretrained=model_config['pretrained']) # remove last layer, fix first layer
        self.output_layer = nn.Linear(1000, 2)

        self.lr = policy_config['lr']
        self.optimizer_name = policy_config['optimizer']
        self.loss_fn = self.configure_loss_fn(policy_config['loss_fn'])
    
    def configure_loss_fn(self, loss_fn):
        if loss_fn == 'CrossEntropy':
            return F.binary_cross_entropy
        
        elif loss_fn == 'ROC_Star':
            return roc_star_loss
        
        else:
            raise NotImplementedError(loss_fn)

    def forward(self, x):
        # resnet
        x = self.resnet(x)
        x = self.output_layer(x)
        
        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        if self.optimizer_name == 'Adam':
            return torch.optim.Adam(self.parameters(), self.lr)

        else:
            raise NotImplementedError(self.optimizer_name)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(y,logits,self.gamma)
        self.log('val_loss', loss)
        return loss
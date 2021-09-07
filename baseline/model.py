import pytorch_lightning as pl
from torchmetrics import (
    MetricCollection,
    Accuracy,
    AUROC,
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
import einops
from common import model_fns
from common.losses import ROCStarLoss

class Reducer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(9, 3, kernel_size=(1, 3), stride=1, padding=(0, 1))

    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x)
    
class LightningG2Net(pl.LightningModule):
    def __init__(self,
                 model_config,
                 optimizer_config,
                 scheduler_config,
                 trainer_config):
        super(LightningG2Net, self).__init__()

        # self.reducer = Reducer()

        self.resnet = self.configure_backbone(model_config.backbone,
                                              model_config.pretrain,
                                              num_classes=512)

        # self.aggregator = nn.LSTM(input_size=512, hidden_size=1024, num_layers=2, batch_first=True)
        self.classification_head = nn.Sequential(nn.Linear(512, 512),
                                                 nn.SiLU(),
                                                 nn.Linear(512, 2))
    
        # hparams
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.model_config = model_config
        self.trainer_config = trainer_config

        self.loss_fn = self.configure_loss_fn()

        # metrics
        self.metrics = MetricCollection([
            Accuracy(num_classes=2, threshold=0.5, dist_sync_on_step=True),
            AUROC(num_classes=2, dist_sync_on_step=True),
        ])

        # aux metrics that we keep track of
        self.prev_epoch_trues = torch.Tensor()
    
    def configure_backbone(self, backbone, pretrained, num_classes):
        if backbone == 'resnet18':
            return resnet18(pretrained=pretrained, num_classes=num_classes) # remove last layer, fix first layer
        elif backbone == 'resnet34':
            return resnet34(pretrained=pretrained, num_classes=num_classes) # remove last layer, fix first layer
        elif backbone == 'resnet50':
            return resnet50(pretrained=pretrained, num_classes=num_classes) # remove last layer, fix first layer     
        elif backbone == 'resnet101':
            return resnet101(pretrained=pretrained, num_classes=num_classes)
        elif backbone == 'googlenet':
            return googlenet(pretrained=pretrained, num_classes=num_classes)
        else:
            raise NotImplementedError(backbone)

    def configure_loss_fn(self):
        return model_fns.configure_loss_fn(self.model_config.loss_fn)

    def configure_optimizers(self):
        n_steps_per_epoch = len(self.train_dataloader())
        return model_fns.configure_optimizers(self.parameters(),
                                              self.optimizer_config,
                                              self.scheduler_config,
                                              self.trainer_config,
                                              n_steps_per_epoch)

    def forward(self, x):
        b, c, t, m = x.shape
        x = self.resnet(x)
        x = self.classification_head(x)

        return x
    
    def on_train_start(self):
        if self.model_config.loss_fn == 'ROC_Star':
            for batch_idx, batch in enumerate(self.train_dataloader()):
                _, targets = batch
                self.loss_fn.epoch_true_acc[batch_idx] = targets
        
            self.loss_fn.on_epoch_end()

    def training_step(self, batch, batch_idx):
        inputs, targets, filename = batch
        inputs = einops.rearrange(inputs, 'b t c f -> b c f t')

        logits = self(inputs)
        loss = self.loss_fn(logits, targets)

        metrics = self.metrics(F.softmax(logits, dim=-1), targets)
        metrics = {f'train/{k}':v for k,v in metrics.items()}

        self.log('train/loss', loss)
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        if self.model_config.loss_fn == 'ROC_Star':
            self.loss_fn.epoch_true_acc[batch_idx] = targets
            self.loss_fn.epoch_pred_acc[batch_idx] = logits
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets, filename = batch
        inputs = einops.rearrange(inputs, 'b t c f -> b c f t')

        logits = self(inputs)
        loss = self.loss_fn(logits, targets)

        metrics = self.metrics(F.softmax(logits, dim=-1), targets)
        metrics = {f'val/{k}':v for k,v in metrics.items()}

        self.log('val/loss', loss)
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        if self.model_config.loss_fn == 'ROC_Star':
            self.loss_fn.epoch_true_acc[batch_idx] = targets
            self.loss_fn.epoch_pred_acc[batch_idx] = logits

        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, targets, filename = batch
        inputs = einops.rearrange(inputs, 'b t c f -> b c f t')

        logits = self(inputs)

        return {'logits': logits, 'targets': targets, 'filename': filename}
    
    def on_train_epoch_end(self):
        if self.model_config.loss_fn == "ROC_Star":
            self.loss_fn.on_epoch_end()
    
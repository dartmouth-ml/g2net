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
from efficientnet_pytorch import EfficientNet
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
        self.conv1d_net = self.create_conv1d_net()
        self.combo_net = self.create_combo_net()
    
        # hparams
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.model_config = model_config
        self.loss_fn = self.configure_loss_fn()

        # metrics
        self.metrics = MetricCollection([
            Accuracy(num_classes=2, threshold=0.5, dist_sync_on_step=True),
            F1(num_classes=2, threshold=0.5, dist_sync_on_step=True),
            AUROC(num_classes=2, dist_sync_on_step=True),
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
        elif backbone == 'efficientnet':
            return EfficientNet.from_pretrained('efficientnet-b0')
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
    def create_combo_net(self):
        combo_net = nn.Sequential(
            nn.Linear(4,2)
        )
        return combo_net

    def create_conv1d_net(self):
        lay1_out = self.conv1d_lout(4096, 4, 4)
        lay2_out = self.conv1d_lout(lay1_out,4,4)
        lay3_out = self.conv1d_lout(lay2_out,4,4)
        lay4_out = self.conv1d_lout(lay3_out,4,4)
        lay5_out = self.conv1d_lout(lay4_out,4,4)
        print(int(lay1_out))
        conv1d_net = nn.Sequential(
            nn.Conv1d(in_channels=3,out_channels=32,kernel_size=4, stride=4),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=4),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=4),
            nn.BatchNorm1d(num_features=128),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=4),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4, stride=4),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(512,2)
        )
        print(conv1d_net)
        return conv1d_net
    
    def conv1d_lout(self, lin, kernel_size, stride):
        return int((lin-(kernel_size-1)-1)/stride + 1)

    def forward(self, x, time_series):
        # resnet
        x1 = self.resnet(x)
        x2 = self.conv1d_net(torch.squeeze(time_series))
        y = self.combo_net(torch.cat((x1, x2),1))
        return y
    
    def on_train_start(self):
        if self.model_config.loss_fn == 'ROC_Star':
            for batch_idx, batch in enumerate(self.train_dataloader()):
                _, targets = batch
                self.loss_fn.epoch_true_acc[batch_idx] = targets
        
            self.loss_fn.on_epoch_end()

    def training_step(self, batch, batch_idx):
        inputs, targets, filename, time_series_data = batch
        logits = self.forward(inputs, time_series_data)
        loss = self.loss_fn(logits, targets.long())

        metrics = self.metrics(F.softmax(logits, dim=-1), targets)
        metrics = {f'train/{k}':v for k,v in metrics.items()}

        self.log('train/loss', loss)
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        if self.model_config.loss_fn == 'ROC_Star':
            self.loss_fn.epoch_true_acc[batch_idx] = targets
            self.loss_fn.epoch_pred_acc[batch_idx] = logits
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets, filename, time_series_data = batch
        logits = self.forward(inputs, time_series_data)
        loss = self.loss_fn(logits, targets.long())

        metrics = self.metrics(F.softmax(logits, dim=-1), targets)
        metrics = {f'val/{k}':v for k,v in metrics.items()}

        self.log('val/loss', loss)
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        if self.model_config.loss_fn == 'ROC_Star':
            self.loss_fn.epoch_true_acc[batch_idx] = targets
            self.loss_fn.epoch_pred_acc[batch_idx] = logits

        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, targets, filename, time_series_data = batch
        logits = self.forward(inputs, time_series_data)

        return {'logits': logits, 'targets': targets, 'filename': filename}
    
    def on_train_epoch_end(self):
        if self.model_config.loss_fn == "ROC_Star":
            self.loss_fn.on_epoch_end()
    
    

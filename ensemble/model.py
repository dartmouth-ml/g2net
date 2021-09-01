import pytorch_lightning as pl
from torch import nn
from torch.nn.functional import softmax
from torch.optim import Adam
from torchmetrics import MetricCollection, Accuracy, AUROC

from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingLR,
)

class EnsembleModel(pl.LightningModule):
    def __init__(self, model_config, optimizer_config, scheduler_config):
        super().__init__()
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.mlp = nn.Sequential(nn.Linear(model_config.n_models, 2*model_config.n_models),
                                 nn.ReLU(),
                                 nn.Linear(2*model_config.n_models, 1))
        self.loss_fn = nn.CrossEntropyLoss()

        # metrics
        self.metrics = MetricCollection([
            Accuracy(num_classes=2, threshold=0.5, dist_sync_on_step=True),
            AUROC(num_classes=2, dist_sync_on_step=True),
        ])
    
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
        
        monitor = scheduler_config.get('monitor', None)

        if monitor is None:
            return {'scheduler': scheduler}
        return {'scheduler': scheduler, 'monitor': scheduler_config.monitor}
    
    def configure_optimizers(self):
        if self.optimizer_config.name == 'Adam':
            optimizer = Adam(self.parameters(), self.optimizer_config.learning_rate)
        else:
            raise NotImplementedError(self.optimizer_config.name)
        
        scheduler_dict = self.configure_lr_schedulers(optimizer, self.scheduler_config)

        if scheduler_dict is None:
            return optimizer
        else:
            return {"optimizer": optimizer, 
                    "lr_scheduler": scheduler_dict}
    
    def forward(self, x):
        return self.mlp(x)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)

        loss = self.loss_fn(logits, targets)

        metrics = self.metrics(softmax(logits, dim=-1), targets)
        metrics = {f'train/{k}':v for k,v in metrics.items()}

        self.log('train/loss', loss)
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)

        loss = self.loss_fn(logits, targets)

        metrics = self.metrics(softmax(logits, dim=-1), targets)
        metrics = {f'val/{k}':v for k,v in metrics.items()}

        self.log('val/loss', loss)
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        return loss

        
    

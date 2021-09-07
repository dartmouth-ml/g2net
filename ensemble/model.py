import pytorch_lightning as pl
from torch import nn
from torch.nn.functional import softmax
from common import model_fns

class EnsembleModel(pl.LightningModule):
    def __init__(self, model_config, optimizer_config, scheduler_config, trainer_config):
        super().__init__()
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.trainer_config = trainer_config

        self.mlp = nn.Sequential(nn.Linear(model_config.n_models, 2*model_config.n_models),
                                 nn.ReLU(),
                                 nn.Linear(2*model_config.n_models, 1))

        self.loss_fn = model_fns.configure_loss_fn(model_config.loss_fn)
        self.metrics = model_fns.configure_metrics()
    
    def configure_optimizers(self):
        return model_fns.configure_optimizers(self.parameters(),
                                              self.optimizer_config,
                                              self.scheduler_config,
                                              self.trainer_config,
                                              n_steps_per_epoch=len(self.train_dataloader()))
    
    def forward(self, x):
        return self.mlp(x)
    
    def step(self, batch, mode):
        inputs, targets, filename = batch
        logits = self(inputs)

        if mode == 'predict':
            return {'logits': logits,
                    'targets': targets,
                    'filename': filename}

        loss = self.loss_fn(logits, targets)

        metrics = self.metrics(softmax(logits, dim=-1), targets)
        metrics = {f'{mode}/{k}':v for k,v in metrics.items()}

        self.log(f'{mode}/loss', loss)
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        return self.step(batch, 'predict')

        
    

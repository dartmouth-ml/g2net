import pytorch_lightning as pl

import torch
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    Linear,
    Conv2d,
    Sequential,
    ReLU,
    Tanh
)
from torch.nn.functional import softmax
from torchvision.models import resnet18
import einops

from common import model_fns

class CRNNModel(pl.LightningModule):
    def __init__(self,
                 model_config,
                 optimizer_config,
                 scheduler_config,
                 trainer_config):
        super().__init__()

        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.trainer_config = trainer_config

        self.collater = Sequential(Conv2d(3, 1, 1, 1, 0), ReLU())
        self.embedder = Sequential(Conv2d(69, 1024, 1, 1, 0), ReLU())
        # self.global_embedder = resnet18(pretrained=False)
        self.classification_head = Sequential(Linear(1024, 1024), ReLU(), Linear(1024, 2))

        self.loss_fn = model_fns.configure_loss_fn(model_config.loss_fn)
        self.metrics = model_fns.configure_metrics()

        encoder_layer = TransformerEncoderLayer(d_model=1024,
                                                nhead=model_config.transformer_nhead,
                                                dim_feedforward=model_config.transformer_dim_feedforward,
                                                batch_first=True)

        self.encoder = TransformerEncoder(encoder_layer,
                                          model_config.transformer_num_layers)
    
    def configure_optimizers(self):
        n_steps_per_epoch = len(self.train_dataloader())
        return model_fns.configure_optimizers(self.parameters(),
                                              self.optimizer_config,
                                              self.scheduler_config,
                                              self.trainer_config,
                                              n_steps_per_epoch)
    
    def forward(self, x):
        b, c, m, t = x.shape

        # first embed the image as a whole
        # x = self.global_embedder(x)

        # combine 3 sites
        x = self.collater(x)
        x = einops.rearrange(x, 'b 1 m t -> b m 1 t')
        x = self.embedder(x)
        x = einops.rearrange(x, 'b m 1 t -> b t m')
        x = self.encoder(x)

        # meanpool all outputs
        x = torch.mean(x, dim=1)
        logits = self.classification_head(x)
        return logits
    
    def step(self, batch, mode):
        inputs, targets, filename = batch
        inputs = einops.rearrange(inputs, 'b t c m -> b c m t')

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

    def training_step(self, batch, *args, **kwargs):
        return self.step(batch, 'train')
    
    def validation_step(self, batch, *args, **kwargs):
        return self.step(batch, 'val')
    
    def predict_step(self, batch, *args, **kwargs):
        return self.step(batch, 'predict')

        
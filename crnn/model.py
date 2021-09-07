import pytorch_lightning as pl

import torch
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    Linear,
    Sequential,
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

        self.stride = model_config.stride
        self.local_embedder = resnet18(pretrained=False)
        self.global_embedder = resnet18(pretrained=False)
        self.global_injector = Sequential(Linear(1000, self.stride), Tanh())
        self.classification_head = Linear(1000, 2)

        self.loss_fn = self.configure_loss_fn()
        self.metrics = model_fns.configure_metrics()

        encoder_layer = TransformerEncoderLayer(d_model=1000,
                                                nhead=model_config.transformer_nhead,
                                                dim_feedforward=model_config.transformer_dim_feedforward,
                                                batch_first=True)

        self.encoder = TransformerEncoder(encoder_layer,
                                          model_config.transformer_num_layers)
    
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
        b, c, m, t = x.shape

        # print(f'x.shape: {x.shape}')
        # first embed the image as a whole
        global_embedding = self.global_embedder(x)
        # print(f'global embedding: {global_embedding.shape}')

        # embed in a sequential manner
        x_seq = einops.rearrange(x, 'b c m (n s) -> (b s) c m n', s=self.stride)
        local_embeddings = self.local_embedder(x_seq)
        local_embeddings = einops.rearrange(local_embeddings, '(b s) d -> b s d', b=b)
        # print(f'local embeddings: {local_embeddings.shape}')

        # before processing sequentially, add global embeddings according to injector weights
        global_embedding_weights = einops.rearrange(self.global_injector(global_embedding),
                                                    'b s -> b s 1')
        # print(f'global embedding weights: {global_embedding_weights.shape}')
        global_embeddings_as_seq = torch.repeat_interleave(einops.rearrange(global_embedding, 'b d -> b 1 d'),
                                                           repeats=self.stride,
                                                           dim=1)
        # print(f'global embedding as seq: {global_embeddings_as_seq.shape}')
        weighted_global_embeddings = torch.mul(global_embeddings_as_seq, global_embedding_weights)
        # print(f'weighted ge: {weighted_global_embeddings.shape}')
        local_embeddings += weighted_global_embeddings
        # print(f"local embeddings after weight: {local_embeddings.shape}")
        sequential_embeddings = self.encoder(local_embeddings)
        # print(f'sequential embeddings: {sequential_embeddings.shape}')

        # take last slice output
        logits = self.classification_head(sequential_embeddings[:, -1, :])
        # print(f'logits shape: {logits.shape}')
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

        
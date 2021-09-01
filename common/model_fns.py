from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingLR,
)

from torch.optim import Adam
from torch.nn import CrossEntropyLoss

def configure_lr_schedulers(optimizer, scheduler_config, trainer_config, n_steps_per_epoch):
        if scheduler_config is None:
            return None
        
        if scheduler_config.name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer)
        
        elif scheduler_config.name == 'StepLR':
            scheduler = StepLR(optimizer, 
                               scheduler_config.step_size,
                               scheduler_config.gamma)
        
        elif scheduler_config.name == 'CosineAnnealing':
            n_steps = n_steps_per_epoch*trainer_config.max_epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=n_steps)

        elif scheduler_config is not None:
            raise NotImplementedError(scheduler_config.name)
        
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': scheduler_config.interval,
        }

        monitor = scheduler_config.get('monitor', None)
        if monitor is not None:
            scheduler_dict['monitor'] = monitor

        return scheduler_dict

def configure_optimizers(params, optimizer_config, scheduler_config, trainer_config, n_steps_per_epoch):
    if optimizer_config.name == 'Adam':
        optimizer = Adam(params=params, lr=optimizer_config.learning_rate)
    else:
        raise NotImplementedError(optimizer_config.name)
    
    scheduler_dict = configure_lr_schedulers(optimizer,
                                             scheduler_config,
                                             trainer_config,
                                             n_steps_per_epoch)

    if scheduler_dict is None:
        return optimizer
    else:
        return {"optimizer": optimizer, 
                "lr_scheduler": scheduler_dict}

def configure_loss_fn(loss_fn):
        if loss_fn == 'CrossEntropy':
            return CrossEntropyLoss(weight=None)
        
        else:
            raise NotImplementedError(loss_fn)
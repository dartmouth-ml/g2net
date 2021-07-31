import torch
from torch import nn
from torch.nn.functional import softmax

class ROCStarLoss(nn.Module):
    def __init__(self, batch_size, dataloader_len, num_classes, delta=1, p=2):
        self.delta = delta
        self.p = p

        self.epoch_true_acc = torch.zeros((dataloader_len, batch_size))
        self.epoch_pred_acc = torch.zeros((dataloader_len, batch_size, num_classes))
        
        self.prev_epoch_true = torch.zeros_like(self.epoch_true_acc)
        self.prev_epoch_pred = softmax(torch.rand_like(self.epoch_pred_acc), dim=-1)

        self.default_gamma = torch.tensor(0.2, dtype=torch.float)
    
    def on_epoch_end(self):
        self.prev_epoch_true = torch.flatten(self.epoch_true_acc)
        self.prev_epoch_pred = torch.flatten(self.epoch_pred_acc)
    
    def forward(self, y_pred, y_true):
        gamma = epoch_update_gamma(y_pred,
                                   y_true,
                                   default_gamma=self.default_gamma,
                                   delta=self.delta)

        loss = roc_star_loss(y_pred,
                             y_true,
                             self.prev_epoch_true,
                             self.prev_epoch_pred,
                             gamma)
        
        return loss

def roc_star_loss(y_pred, _y_true, _epoch_true, epoch_pred, gamma):
    """
    Nearly direct loss function for AUC.
    See article,
    C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
    https://github.com/iridiumblue/articles/blob/master/roc_star.md
        _y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
        y_pred: `Tensor` . Predictions.
        gamma  : `Float` Gamma, as derived from last epoch.
        _epoch_true: `Tensor`.  Targets (labels) from last epoch.
        epoch_pred : `Tensor`.  Predicions from last epoch.
    """
    #convert labels to boolean
    y_true = (_y_true>=0.50)
    epoch_true = (_epoch_true>=0.50)

    # if batch is either all true or false return small random stub value.
    if torch.sum(y_true)==0 or torch.sum(y_true) == y_true.shape[0]:
        return torch.sum(y_pred)*1e-8

    pos = y_pred[y_true]
    neg = y_pred[~y_true]

    epoch_pos = epoch_pred[epoch_true]
    epoch_neg = epoch_pred[~epoch_true]

    # Take random subsamples of the training set, both positive and negative.
    max_pos = 1000 # Max number of positive training samples
    max_neg = 1000 # Max number of positive training samples
    cap_pos = epoch_pos.shape[0]
    cap_neg = epoch_neg.shape[0]
    epoch_pos = epoch_pos[torch.rand_like(epoch_pos) < max_pos/cap_pos]
    epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg/cap_pos]

    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]

    # sum positive batch elements agaionst (subsampled) negative elements
    if ln_pos>0 :
        pos_expand = pos.view(-1,1).expand(-1,epoch_neg.shape[0]).reshape(-1)
        neg_expand = epoch_neg.repeat(ln_pos)

        diff2 = neg_expand - pos_expand + gamma
        l2 = diff2[diff2>0]
        m2 = l2 * l2
    else:
        m2 = torch.tensor([0], dtype=torch.float).cuda()

    # Similarly, compare negative batch elements against (subsampled) positive elements
    if ln_neg>0 :
        pos_expand = epoch_pos.view(-1,1).expand(-1, ln_neg).reshape(-1)
        neg_expand = neg.repeat(epoch_pos.shape[0])

        diff3 = neg_expand - pos_expand + gamma
        l3 = diff3[diff3>0]
        m3 = l3*l3
    else:
        m3 = torch.tensor([0], dtype=torch.float).cuda()

    if (torch.sum(m2)+torch.sum(m3))!=0 :
        res2 = torch.sum(m2)/max_pos+torch.sum(m3)/max_neg
    else:
        res2 = torch.sum(m2)+torch.sum(m3)

    res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

    return res2

def epoch_update_gamma(y_pred, y_true, default_gamma, delta=2, subsample_size=2000):
    """
    Calculate gamma from last epoch's targets and predictions.
    Gamma is updated at the end of each epoch.
    y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
    y_pred: `Tensor` . Predictions.
    """
    pos = y_pred[y_true==1]
    neg = y_pred[y_true==0]

    # subsample the training set for performance
    cap_pos = pos.shape[0]
    cap_neg = neg.shape[0]

    pos = pos[torch.rand_like(pos) < subsample_size/cap_pos]
    neg = neg[torch.rand_like(neg) < subsample_size/cap_neg]

    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]

    pos_expand = pos.view(-1,1).expand(-1,ln_neg).reshape(-1)
    neg_expand = neg.repeat(ln_pos)

    diff = neg_expand - pos_expand
    Lp = diff[diff>0] # because we're taking positive diffs, we got pos and neg flipped.
    ln_Lp = Lp.shape[0]-1

    diff_neg = -1.0 * diff[diff<0]
    diff_neg = diff_neg.sort()[0]

    ln_neg = diff_neg.shape[0]-1
    ln_neg = max([ln_neg, 0])

    left_wing = int(ln_Lp*delta)
    left_wing = max([0,left_wing])
    left_wing = min([ln_neg,left_wing])

    if diff_neg.shape[0] > 0 :
       gamma = diff_neg[left_wing]
    else:
       gamma = default_gamma
    
    return gamma
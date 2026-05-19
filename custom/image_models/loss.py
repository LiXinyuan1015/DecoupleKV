import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, x, y, return_batch_loss = False):
        log_probs = F.log_softmax(x, dim=-1)
        if return_batch_loss:
            loss = -log_probs.gather(-1, y[..., None])
        else:
            loss = -log_probs.gather(-1, y[..., None]).mean()
        return loss
    
class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smothing, num_labels):
        super().__init__()
        self.label_smoothing = 1 - label_smothing
        self.num_labels = num_labels

    def forward(self, x, y):
        y = nn.functional.one_hot(y, self.num_labels)
        prob_margin = 1 / (self.num_labels - 1)
        y = prob_margin + self.label_smoothing * y
        
        loss = F.kl_div(F.log_softmax(x, -1), y, reduction='batchmean')
        return loss
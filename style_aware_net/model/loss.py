import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import TripletMarginLoss

def TripletLoss(anc_projs, pos_projs, neg_projs, loss_weight, margin=2):
    total_loss = 0.0
    for i, (anc, pos, neg) in enumerate(zip(anc_projs, pos_projs, neg_projs)):
        loss = torch.nn.TripletMarginLoss(margin=margin, reduction='none')(anc, pos, neg)
        loss = loss_weight.T[i] * loss
        total_loss += torch.mean(loss)
    return total_loss
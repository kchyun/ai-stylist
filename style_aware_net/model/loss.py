import torch
from torch import Tensor
from torch.nn import TripletMarginLoss

def TripletLoss(anchor, pos, neg):
    return TripletMarginLoss(margin=5)(anchor, pos, neg)
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

def compute_loss(anc_projs, pos_projs, neg_projs, loss_weight):
    total_loss = 0.0
    for i, (anc, pos, negs) in enumerate(zip(anc_projs, pos_projs, neg_projs)):
        # (B, )
        loss = info_nce(anc, pos, negs, reduction='none')
        loss = loss_weight.T[i] * loss
        total_loss += torch.mean(loss)
    return total_loss


def info_nce(anc, pos, negs, temparature=0.1, reduction='mean'):
    # Normalize all embeds
    anc = F.normalize(anc, dim=-1)
    pos = F.normalize(pos, dim=-1)
    negs = F.normalize(negs, dim=-1)

    # (B, E), (B, E) -> (B, 1)
    pos_logit = torch.sum(anc * pos, dim=1, keepdim=True)
    # (B, E, 1), (B, E, N_S) -> (B, E, N_S) -> (B, N_S)
    neg_logits = torch.sum(anc.unsqueeze(1) * negs, dim=2)
    # (B, E, 1), (B, E, N_S) -> (B, N_S + 1)
    logits = torch.cat([pos_logit, neg_logits], dim=1)

    # We will use torch's cross entorpy function to compute log loss.
    # (B, N_S + 1)
    logits = torch.softmax(logits / temparature, dim=1)
    # (B, )
    labels = torch.zeros(len(logits), dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, labels, reduction=reduction)


if __name__ == '__main__':
    B, E, N_S = 2, 4, 2
    anc = torch.rand(B, E)
    pos = torch.rand(B, E)
    negs = torch.rand(B, N_S, E)
    print(anc)
    print(negs)
    info_nce(anc, pos, negs, temparature=0.1, reduction='mean')

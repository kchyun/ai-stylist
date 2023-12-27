import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from PIL.Image import Image


class FashionRecommender():
    def __init__(self, model, embed_generator, device):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        self.model = model
        self.model.eval()

        self.embed_generator = embed_generator

        self.model.to(self.device)
        self.embed_generator.to(self.device)

    @torch.no_grad()
    def single_infer(self, top, bottom):
        # import pdb; pdb.set_trace()
        # top이 한개씩 들어온 것인가...?
        top_proj = self.model(torch.Tensor([t['embed'] for t in top]).to(self.device))
        bottom_proj = self.model(torch.Tensor([b['embed'] for b in bottom]).to(self.device))
        bottom_proj = torch.stack([b for b in bottom_proj])
        print("Projection done")
        
        # score = torch.mm(torch.nn.functional.normalize(top_proj, dim=-1), torch.nn.functional.normalize(bottom_proj, dim = -1).mT)
        
        # import pdb; pdb.set_trace()
        score_lst = []
        for i, t in enumerate(top):
            score = torch.nn.PairwiseDistance(p=2)(torch.Tensor(top_proj[i]), bottom_proj)
            for j, b in enumerate(bottom):
                score_lst.append({'top_id' : t['id'], 'bottom_id' : b['id'], 'score' : score[j]})
        
        score_lst = sorted(score_lst, key=lambda x: x['score'])
        #score = torch.argmax(torch.nn.CosineSimilarity(dim=1)(top_proj, bottom_proj), dim=1)
        
        return score_lst

    
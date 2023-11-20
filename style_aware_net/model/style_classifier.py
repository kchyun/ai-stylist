import sys
from torchmetrics.functional import pairwise_cosine_similarity
import torch
from typing import List

class StyleClassifier():
    def __init__(self, 
                 embed_generator,
                 styles: List[str]):
        
        self.styles = styles
        self.embed_generator = embed_generator
        self.prompt_embeddings = torch.stack([torch.Tensor(self.embed_generator.text2embed("a photo of {} style clothes".format(s))) for s in styles])
        
    def forward(self, anc, pos):
        
        # import pdb; pdb.set_trace()
        
        pos_similarity = pairwise_cosine_similarity(pos, self.prompt_embeddings.squeeze(1))
        anc_similarity = pairwise_cosine_similarity(anc, self.prompt_embeddings.squeeze(1))

        
        # anc_similarity = torch.nn.functional.softmax(anc_similarity, dim=1)
        # pos_similarity = torch.nn.functional.softmax(pos_similarity, dim=1)
        loss_weight = torch.min(torch.stack([anc_similarity, pos_similarity]), 0).values
        return torch.nn.functional.softmax(loss_weight, dim=1)

    def idx2stl(self, idx):
        return self.styles[idx]
        
    def stl2idx(self, stl):
        return self.styles.index(stl)
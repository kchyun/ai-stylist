import sys
from torchmetrics.functional import pairwise_cosine_similarity
import torch
from typing import List

class StyleClassifier():
    def __init__(self, 
                 embed_generator,
                 styles: List[str]):
        
        self.styles = styles
        self.temperature = 0.5
        self.embed_generator = embed_generator
        self.prompt_embeddings = torch.stack([torch.Tensor(self.embed_generator.text2embed("a photo of {} style clothes".format(s))) for s in styles])

    
    @torch.no_grad()
    def forward(self, anc, category, device):
        # category는 number로 줄게용
        anc_similarity = pairwise_cosine_similarity(torch.Tensor(anc).reshape(1, -1).to(device), self.prompt_embeddings.squeeze(1).to(device)).view(-1)
        return anc_similarity[category]

    def idx2stl(self, idx):
        return self.styles[idx]
        
    def stl2idx(self, stl):
        return self.styles.index(stl)
import torch
import torch.nn as nn

class FashionMLP(nn.Module):
    def __init__(self, embed_size=512, ModelArgs=None):
        super().__init__()
        embed_size = embed_size

        self.fc_layer = nn.Sequential(
            nn.Linear(embed_size * 2, 1024),
            nn.GELU(),
            nn.Linear(1024, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, 1)
            )
        self.config = ModelArgs
    
    def forward(self, i, j):
        x = torch.concat([i, j], dim=1)
        output = self.fc_layer(x)
        return output


class FashionSIAMESE(nn.Module):
    def __init__(self, ModelArgs):
        super().__init__()
        embed_size = ModelArgs.embed_size

        self.model = nn.Sequential(
            nn.Linear(embed_size, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            )
        self.top = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )
        self.bottom = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )

        self.config = ModelArgs

    def forward(self, x, y, z):
        anc = self.top(self.model(x))
        pos = self.bottom(self.model(y))
        neg = self.bottom(self.model(z))
        return anc, pos, neg
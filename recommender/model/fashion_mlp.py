import torch
import torch.nn as nn

class FashionMLP(nn.Module):
    def __init__(self, ModelArgs):
        super().__init__()
        embed_size = ModelArgs.embed_size

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

        self.fc_layer = nn.Sequential(
            nn.Linear(embed_size, 1024),
            nn.GELU(),
            nn.Linear(1024, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, 512)
        )

        self.config = ModelArgs
    
    def forward(self, x):
        output = self.fc_layer(x)
        return output
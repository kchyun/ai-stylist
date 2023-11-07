import torch
import torch.nn as nn

class FashionMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        embed_size = config.embed_size

        self.fc_for_top = nn.Linear(embed_size, embed_size)
        self.fc_for_bottom = nn.Linear(embed_size, embed_size)

        self.fc_for_all1 = nn.Linear(2*embed_size, embed_size)
        self.fc_for_all2 = nn.Linear(embed_size, 1)

        self.relu = nn.ReLU()
        self.config = config
    
    def forward(self, top_embed, bottom_embed, negative_bottom_embed):
        
        if self.config.use_linear:
            top_embed = self.relu(self.fc_for_top(top_embed))
            bottom_embed = self.relu(self.fc_for_bottom(bottom_embed))
            negative_bottom_embed = self.relu(self.fc_for_bottom(negative_bottom_embed))

        concat_positive = torch.cat([top_embed, bottom_embed], dim=1)
        output_positive = self.relu(self.fc_for_all1(concat_positive))
        output_positive = self.fc_for_all2(output_positive)

        concat_negative = torch.cat([top_embed, negative_bottom_embed], dim = 1)
        output_negative = self.relu(self.fc_for_all1(concat_negative))
        output_negative = self.fc_for_all2(output_negative)


        return output_positive, output_negative
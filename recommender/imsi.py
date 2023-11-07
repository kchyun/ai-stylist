import json
import random
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

PAIR_PATH = '/Users/owj0421/Desktop/Projects/2023/ai_stylist/data/FashionVCdata/gnd_top_bottom_pairs.csv'
TOP_EMBED_PATH = '/Users/owj0421/Desktop/Projects/2023/ai_stylist/data/FashionVCdata/top_embeds.csv'
BOTTOM_EMBED_PATH = '/Users/owj0421/Desktop/Projects/2023/ai_stylist/data/FashionVCdata/bottom_embeds.csv'


class FashionDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.pairs = self.get_pairs(PAIR_PATH)
        self.top_embeds = self.get_embeds(TOP_EMBED_PATH)
        self.bottom_embeds = self.get_embeds(BOTTOM_EMBED_PATH)
        
        self.positive_pairs = defaultdict(set)
        for i, (top_idx, bottom_idx) in self.pairs.iterrows():
            self.positive_pairs[str(top_idx)].add(str(bottom_idx))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        top_idx, bottom_idx = map(str, self.pairs[idx])
        negative_idx = self._get_negative_bottom_idx(top_idx)
        return {
            'top' : LongTensor(self.top_embeds[top_idx]),
            'bottom' : LongTensor(self.bottom_embeds[bottom_idx]),
            'negative_bottom' : LongTensor(self.bottom_embeds[negative_idx])
            }
        
    def _get_negative_bottom_idx(self, top_idx):
        negative_idx = str(random.randint(0, len(self.bottom_embeds) - 1))
        cnt = 0
        while negative_idx in self.positive_pairs[top_idx] and cnt < 5:
            negative_idx = str(random.randint(0, len(self.bottom_embeds)))
            cnt += 1
        return negative_idx

    def get_pairs(self, path):
        return pd.read_csv(path, header=None, names=['top', 'bottom'])

    def get_embeds(self, path):
        embed_df = pd.read_csv(path)
        return OrderedDict({str(embed_data['id']): embed_data['embed'] for i, embed_data in embed_df.iterrows()})

class FashionModel(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.fc_for_top = nn.Linear(embed_size, embed_size)
        self.fc_for_bottom = nn.Linear(embed_size, embed_size)

        self.fc_for_all1 = nn.Linear(2*embed_size, embed_size)
        self.fc_for_all2 = nn.Linear(embed_size, 1)
        self.fc_for_last = nn.Sigmoid()

        self.relu = nn.ReLU()

    def forward(self, top_embed, bottom_embed):
        output_top = self.relu(self.fc_for_top(top_embed))
        output_bottom = self.relu(self.fc_for_bottom(bottom_embed))

        concat_all = torch.cat([output_top, output_bottom], dim=1)
        output_all = self.relu(self.fc_for_all1(concat_all))
        output_all = self.fc_for_last(self.fc_for_all2(output_all))
        return output_all

print("실행")
tr_dataset = FashionDataset()
tr_dataloader = DataLoader(dataset=tr_dataset, batch_size=32, shuffle=True)


epoch = 10
embed_size = 512
model = FashionModel(embed_size)
criterion = nn.BCELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.00001)

for i in range(epoch):
    running_loss = 0.0
    for j, data in enumerate(tr_dataloader):
        optim.zero_grad()
        labels = data["label"]

        labels = labels.unsqueeze(-1)
        pred_pos = model(data['top'], data['bottom'])
        pred_neg = model(data['top'], data['negative_bottom'])

        loss = - (pred_pos - pred_neg).sigmoid().log().sum()
        loss.backward()
        optim.step()

        running_loss += loss.item()
        if j % 100 == 99:
            print(f'[{epoch + 1}, {j + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0




import os
import argparse
import torch
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.optim import Adam, lr_scheduler
from model.loss import PairwiseRankingLoss, TripletLoss
from model.fashion_mlp import FashionMLP, FashionSIAMESE


class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, optimizer, scheduler, device, TrainingArgs):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.scheduler = scheduler
        self.device = device
        self.TrainingArgs = TrainingArgs

        self.best_model_state = None
        self.best_optimizer_state = None


    def train(self):
        lowest_loss = np.inf
        for epoch in range(self.TrainingArgs.n_epochs):
            train_loss = self._train(self.train_dataloader, epoch)
            valid_loss = self._validate(self.valid_dataloader, epoch)
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                self.best_model_state = deepcopy(self.model.state_dict())
                self.best_optimizer_state = deepcopy(self.optimizer.state_dict())

    def _train(self, dataloader, epoch):
        self.model.train()

        epoch_iterator = tqdm(dataloader)

        losses = 0.0
        for iter, batch in enumerate(epoch_iterator, start=1):
            self.optimizer.zero_grad()

            if isinstance(self.model, FashionMLP):
                source_embed, pos_embed, neg_embeds = batch
                pos = self.model(source_embed.to(self.device), pos_embed.to(self.device))
                # (batch_size, negative sample size, 1) to (negative sample size, 1)
                neg = torch.mean(torch.stack([self.model(source_embed.to(self.device), neg_embed.to(self.device)) for neg_embed in neg_embeds]), dim=0) 
                loss = PairwiseRankingLoss(pos, neg)
            elif isinstance(self.model, FashionSIAMESE):
                source_embed, pos_embed, neg_embed, source_embed_sim, pos_embed_sim, neg_embed_sim = batch
                loss = TripletLoss(source_embed, pos_embed, neg_embed) + TripletLoss(source_embed_sim, pos_embed_sim, neg_embed_sim)
            else:
                raise ValueError("Unknown Model type")
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            losses += loss.item()
            epoch_iterator.set_description(
                'Train | Epoch: {:03}/{:03} | loss: {:.5f}'.format(epoch + 1, self.TrainingArgs.n_epochs, losses / iter)
                )

        return losses / iter


    @torch.no_grad()
    def _validate(self, dataloader, epoch):
        self.model.eval()

        epoch_iterator = tqdm(dataloader)

        losses = 0.0

        for iter, batch in enumerate(epoch_iterator, start=1):
            if isinstance(self.model, FashionMLP):
                source_embed, pos_embed, neg_embeds = batch
                pos = self.model(source_embed.to(self.device), pos_embed.to(self.device))
                neg = torch.mean(torch.stack([self.model(source_embed.to(self.device), neg_embed.to(self.device)) for neg_embed in neg_embeds]), dim=0)
                loss = PairwiseRankingLoss(pos, neg)
            elif isinstance(self.model, FashionSIAMESE):
                source_embed, pos_embed, neg_embed, source_embed_sim, pos_embed_sim, neg_embed_sim = batch
                loss = TripletLoss(source_embed, pos_embed, neg_embed) + TripletLoss(source_embed_sim, pos_embed_sim, neg_embed_sim)
            else:
                raise ValueError("Unknown Model type")

            losses += loss.item()

            epoch_iterator.set_description(
                'Valid | Epoch: {:03}/{:03} | loss: {:.5f}'.format(epoch + 1, self.TrainingArgs.n_epochs, losses / iter)
                )

        return losses / iter

    def save(self, model_path, model_name, best_model: bool=True):
        model_path = os.path.join(model_path, f'{model_name}.pth')
        if best_model:
            torch.save(self.best_model_state, model_path)
        else:
            torch.save(self.model.state_dict(), model_path)
        print(f'Model successfully saved at {model_path}')


    def load(self, model_path, model_name):
        model_path = os.path.join(model_path, f'{model_name}.pth')
        self.model.load_state_dict(torch.load(model_path))
        print(f'Model successfully loaded from {model_path}')
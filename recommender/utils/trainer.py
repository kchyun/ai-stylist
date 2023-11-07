import argparse
import torch
import tqdm
import numpy as np

from torch.optim import Adam, lr_scheduler
from model.loss import PairwiseRankingLoss


class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, optimizer, scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.scheduler = scheduler
        self.config = config

    def train(self):
        lowest_loss = np.inf
        best_model = None

        for epoch in range(config.n_epochs):
            train_loss = self._train(self.train_dataloader, epoch, self.config)
            valid_loss = self._validate(self.valid_dataloader, self.config)

            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        return best_model

    def _train(self, dataloader, epoch, config):
        
        self.model.train()

        epoch_iterator = tqdm(
            dataloader, desc='epoch X/X, global: XXX/XXX, tr_loss: XXX'
        )
        epoch = epoch + 1

        for idx, batch in enumerate(epoch_iterator):
            
            positive_top_embed, positive_bottom_embed = batch['positive_pair']
            _, negative_bottom_embed = batch['negative_pair']
            self.optimizer.zero_grad()

            self.model(positive_top_embed, positive_bottom_embed, negative_bottom_embed)

            loss =  PairwiseRankingLoss()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            global_step += 1

            
        )


    
    def _validate(self, data_loader, epoch, config):
        self.model.eval()

        with torch.no_grad():
            positive



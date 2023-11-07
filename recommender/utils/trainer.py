import argparse
import torch
import numpy as np

from tqdm import tqdm
from copy import deepcopy
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

        self.best_model = None
        self.best_optimizer = None

    def train(self):
        lowest_loss = np.inf

        for epoch in range(self.config.n_epochs):
            train_loss = self._train(self.train_dataloader, epoch)
            valid_loss = self._validate(self.valid_dataloader)

            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                self.best_model = deepcopy(self.model.state_dict())
                self.best_optimizer = deepcopy(self.optimizer.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch + 1,
                self.config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))


    def _train(self, dataloader, epoch):
        
        self.model.train()

        epoch_iterator = tqdm(
            dataloader, desc='epoch X/X, tr_loss: XXX'
        )
        epoch = epoch + 1
        sum_loss = 0.0
        mean_loss = 0.0
        for idx, batch in enumerate(epoch_iterator):
            
            positive_top_embed, positive_bottom_embed = batch['positive_pair']
            _, negative_bottom_embed = batch['negative_pair']
            self.optimizer.zero_grad()

            output_positive, output_negative = self.model(positive_top_embed, positive_bottom_embed, negative_bottom_embed)

            loss = PairwiseRankingLoss(output_positive, output_negative)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            sum_loss += loss.item()
            mean_loss = sum_loss / (idx + 1)

            epoch_iterator.set_description(
                'epoch: {}/{}, tr_loss: {:.3f}'.format(
                    epoch, self.config.n_epochs,
                    mean_loss
                )
            )
        
        return mean_loss


    
    def _validate(self, dataloader):
        self.model.eval()
        
        valid_iterator = tqdm(
            dataloader, desc='valid_loss: XXX'
        )

        with torch.no_grad():
            sum_loss = 0.0
            mean_loss = 0.0
            for idx, batch in enumerate(valid_iterator):
                positive_top_embed, positive_bottom_embed = batch['positive_pair']
                _, negative_bottom_embed = batch['negative_pair']

                output_positive, output_negative = self.model(positive_top_embed, positive_bottom_embed, negative_bottom_embed)

                loss = PairwiseRankingLoss(output_positive, output_negative)
                
                sum_loss += loss.item()
                mean_loss = sum_loss / (idx + 1)

                valid_iterator.set_description(
                    'valid_loss: {:.3f}'.format(mean_loss)
                )
                
            return mean_loss


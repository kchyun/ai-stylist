import os
import argparse
import torch
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.optim import Adam, lr_scheduler
from model.loss import TripletLoss


class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, optimizer, scheduler, style_classifier, device, args):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.scheduler = scheduler
        self.device = device
        self.args = args

        self.style_classifier = style_classifier

        self.best_model_state = None
        self.best_optimizer_state = None

    def train(self):
        lowest_loss = np.inf
        for epoch in range(self.args.n_epochs):
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

            anc, pos, negs = batch

            loss_weight = self.style_classifier.forward(anc, pos)
            loss_weight = loss_weight.to(self.device)

            anc_projs = self.model(anc.to(self.device))
            pos_projs = self.model(pos.to(self.device))
            # Shape of negs: B, N_S, D
            # (B*N_S, D)
            negsv = negs.view(-1, negs.shape[-1])
            # N_C * (B*N_S, E)
            neg_projs = self.model(negsv.to(self.device))
            # N_C * (B, N_S, E)
            neg_projs = [neg_proj.view(negs.shape[0], negs.shape[1], -1) for neg_proj in neg_projs]
            # N_C * (B, E)
            neg_projs = [torch.mean(neg_proj, dim=1) for neg_proj in neg_projs]

            loss = TripletLoss(anc_projs, pos_projs, neg_projs, loss_weight)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            losses += loss.item()

            epoch_iterator.set_description(
                'Train | Epoch: {:03}/{:03} | loss: {:.5f}'.format(epoch + 1, self.args.n_epochs, losses / iter))

        return losses / iter


    @torch.no_grad()
    def _validate(self, dataloader, epoch):
        self.model.eval()
        epoch_iterator = tqdm(dataloader)
        losses = 0.0
        for iter, batch in enumerate(epoch_iterator, start=1):

            anc, pos, negs = batch

            loss_weight = self.style_classifier.forward(anc, pos)
            loss_weight = loss_weight.to(self.device)

            anc_projs = self.model(anc.to(self.device))
            pos_projs = self.model(pos.to(self.device))
            # Shape of negs: B, N_S, D
            # (B*N_S, D)
            negsv = negs.view(-1, negs.shape[-1])
            # N_C * (B*N_S, E)
            neg_projs = self.model(negsv.to(self.device))
            # N_C * (B, N_S, E)
            neg_projs = [neg_proj.view(negs.shape[0], negs.shape[1], -1) for neg_proj in neg_projs]
            # N_C * (B, E)
            neg_projs = [torch.mean(neg_proj, dim=1) for neg_proj in neg_projs]

            loss = TripletLoss(anc_projs, pos_projs, neg_projs, loss_weight)
            
            losses += loss.item()

            epoch_iterator.set_description(
                'Valid | Epoch: {:03}/{:03} | loss: {:.5f}'.format(epoch + 1, self.args.n_epochs, losses / iter))

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
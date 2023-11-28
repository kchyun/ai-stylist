import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from copy import deepcopy
from torch.optim import Adam, lr_scheduler
from model.loss import TripletLoss
from transformers import  CLIPVisionModelWithProjection, CLIPProcessor

class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, optimizer, scheduler, style_classifier, device, args):
        self.embed_model = CLIPVisionModelWithProjection.from_pretrained('patrickjohncyh/fashion-clip').to(device)
        for param in self.embed_model.parameters():
            param.requires_grad = False
        self.embed_model.eval()
        self.processor = CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip')

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

            if epoch % self.args.save_every == 0:
                model_name = f'{epoch}_{valid_loss:.3f}'
                self.save(self.args.save_path, model_name)

    """Modification"""
    def _filter_negatives(self, pos_embed, neg_embeds, threshold):

        # pos_embed shape: (B, E), neg_embeds shape: (B, N_S, E)
        B, N_S, E = neg_embeds.shape
        pos_embed = pos_embed.unsqueeze(1)  # (B, 1, E)
        similarities = F.cosine_similarity(pos_embed, neg_embeds, dim=2)  # (B, N_S)

        # Mask for negatives above the threshold
        mask = similarities >= threshold

        # If all negatives are below the threshold, keep the one with the lowest similarity
        all_below_threshold = mask.sum(dim=1) == 0
        lowest_sim_indices = similarities.argmin(dim=1)
        mask[torch.arange(B)[all_below_threshold], lowest_sim_indices[all_below_threshold]] = True

        # Filter out embeddings below threshold
        filtered_neg_embeds = neg_embeds[mask.unsqueeze(-1).expand(-1, -1, E)].view(B, -1, E)

        return filtered_neg_embeds, mask.view(B, N_S)

    def _train(self, dataloader, epoch):
        self.model.train()
        epoch_iterator = tqdm(dataloader)
        losses = 0.0
        threshold = 0.5
        for iter, batch in enumerate(epoch_iterator, start=1):
            self.optimizer.zero_grad()

            anc_img, pos_img, neg_imgs = batch

            anc_img = self.processor(images=anc_img, return_tensors="pt", padding=True)
            anc_embed = self.embed_model(**anc_img.to(self.device)).image_embeds.detach()
            anc_projs = self.model(anc_embed)

            pos_img = self.processor(images=pos_img, return_tensors="pt", padding=True)
            pos_embed = self.embed_model(**pos_img.to(self.device)).image_embeds.detach()
            pos_projs = self.model(pos_embed)
            
            # Shape of negs: (B, N_S, H, W, C)
            # (B*N_S, H, W, C)

            neg_imgs = neg_imgs.view(-1, neg_imgs.size(2), neg_imgs.size(3), neg_imgs.size(4))
            neg_imgs = self.processor(images=neg_imgs, return_tensors="pt", padding=True)
            neg_embeds = self.embed_model(**neg_imgs.to(self.device)).image_embeds.detach()
            """Modification"""
            neg_embeds_filtered, mask = self._filter_negatives(pos_embed, neg_embeds, threshold)
            neg_imgs_filtered = neg_imgs[mask].view(B, -1, neg_imgs.size(2), neg_imgs.size(3), neg_imgs.size(4))
            B = neg_imgs_filtered.size(0)
            N_S = neg_imgs_filtered.size(1)
            # N_C * (B*N_S, E)
            neg_projs = self.model(neg_embeds_filtered)
            
            # N_C * (B, N_S, E)
            neg_projs = [neg_proj.view(B, N_S, -1) for neg_proj in neg_projs]
            # N_C * (B, E)
            neg_projs = [torch.mean(neg_proj, dim=1) for neg_proj in neg_projs]

            # loss 가중치 결정
            style_logits = self.style_classifier.forward(anc_embed, pos_embed, self.device)
            loss_weight = style_logits  # 뭐 어떤 threshold 혹은 기타 처리하자 나중에...
            loss_weight = loss_weight.to(self.device)

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
        threshold = 0.5
        for iter, batch in enumerate(epoch_iterator, start=1):
            anc_img, pos_img, neg_imgs = batch

            anc_img = self.processor(images=anc_img, return_tensors="pt", padding=True)
            anc_embed = self.embed_model(**anc_img.to(self.device)).image_embeds.detach()
            anc_projs = self.model(anc_embed)

            pos_img = self.processor(images=pos_img, return_tensors="pt", padding=True)
            pos_embed = self.embed_model(**pos_img.to(self.device)).image_embeds.detach()
            pos_projs = self.model(pos_embed)

            neg_imgs = neg_imgs.view(-1, neg_imgs.size(2), neg_imgs.size(3), neg_imgs.size(4))
            neg_imgs = self.processor(images=neg_imgs, return_tensors="pt", padding=True)
            neg_embeds = self.embed_model(**neg_imgs.to(self.device)).image_embeds.detach()
            """Modification"""
            neg_embeds_filtered, mask = self._filter_negatives(pos_embed, neg_embeds, threshold)
            neg_imgs_filtered = neg_imgs[mask].view(B, -1, neg_imgs.size(2), neg_imgs.size(3), neg_imgs.size(4))
            B = neg_imgs_filtered.size(0)
            N_S = neg_imgs_filtered.size(1)
            
            neg_projs = self.model(neg_embeds_filtered)
            neg_projs = [neg_proj.view(B, N_S, -1) for neg_proj in neg_projs]
            neg_projs = [torch.mean(neg_proj, dim=1) for neg_proj in neg_projs]

            style_logits = self.style_classifier.forward(anc_embed, pos_embed, self.device)
            loss_weight = style_logits
            loss_weight = loss_weight.to(self.device)

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
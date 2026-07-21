#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pytorch_lightning as pl
from torch import nn


class DiscreteGesturesModule(pl.LightningModule):
    def __init__(
        self,
        network,
        learning_rate=1e-3
    ):
        super().__init__()

        self.network = network
        self.learning_rate = learning_rate

        # Multi-label binary classification loss
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, emg):
        return self.network(emg)

    def training_step(self, batch, batch_idx):

        emg = batch["emg"]              # batch x 32 x time
        targets = batch["targets"]      # batch x 7 x time

        preds = self.network(emg)          # [B, 7, T_out]
        # Align time dimension: crop to shorter of the two
        T = min(preds.shape[-1], targets.shape[-1])
        preds = preds[:, :, :T]
        targets = targets[:, :, :T].float()

        loss = self.loss_fn(preds, targets)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True
        )

        return loss

    def validation_step(self, batch, batch_idx):

        emg = batch["emg"]
        targets = batch["targets"]

        preds = self.network(emg)          # [B, 7, T_out]
        # Align time dimension: crop to shorter of the two
        T = min(preds.shape[-1], targets.shape[-1])
        preds = preds[:, :, :T]
        targets = targets[:, :, :T].float()

        loss = self.loss_fn(preds, targets)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True
        )

        return loss

    def test_step(self, batch, batch_idx):

        emg = batch["emg"]
        targets = batch["targets"]

        preds = self.network(emg)          # [B, 7, T_out]
        # Align time dimension: crop to shorter of the two
        T = min(preds.shape[-1], targets.shape[-1])
        preds = preds[:, :, :T]
        targets = targets[:, :, :T].float()

        loss = self.loss_fn(preds, targets)

        self.log(
            "test_loss",
            loss,
            sync_dist=True
        )

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )

        return optimizer


# In[ ]:





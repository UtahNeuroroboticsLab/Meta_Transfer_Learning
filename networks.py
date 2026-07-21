#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import numpy as np

class ReinhardCompression(nn.Module):
    def __init__(self, range=64.0, midpoint=32.0):
        super().__init__()
        self.range = range
        self.midpoint = midpoint

    def forward(self, x):
        return self.range * x / (self.midpoint + torch.abs(x))


class DiscreteGesturesArchitecture(nn.Module):
    def __init__(
        self,
        input_channels=32,      # changed from Meta's 16 to your 32
        conv_output_channels=128,
        kernel_width=21,
        stride=10,
        lstm_hidden_size=128,
        lstm_num_layers=3,
        output_channels=7       # your 7 gestures
    ):
        super().__init__()

        self.compression = ReinhardCompression(range=64.0, midpoint=32.0)

        self.conv_layer = nn.Conv1d(
            input_channels,
            conv_output_channels,
            kernel_size=kernel_width,
            stride=stride,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.post_conv_layer_norm = nn.LayerNorm(conv_output_channels)

        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.1,
        )

        self.post_lstm_layer_norm = nn.LayerNorm(lstm_hidden_size)
        self.projection = nn.Linear(lstm_hidden_size, output_channels)

    def forward(self, x):
        # input x shape: batch x channels x time

        x = self.compression(x)

        x = self.conv_layer(x)
        x = self.relu(x)
        x = self.dropout(x)

        # batch x channels x time -> batch x time x channels
        x = x.transpose(1, 2)
        x = self.post_conv_layer_norm(x)

        x, _ = self.lstm(x)
        x = self.post_lstm_layer_norm(x)

        x = self.projection(x)

        # batch x time x gestures -> batch x gestures x time
        x = x.permute(0, 2, 1)

        return x


# In[ ]:





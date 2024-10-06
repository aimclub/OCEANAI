#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Архитектуры текстовых моделей для Torch
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key):
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = F.softmax(scores, dim=-1)
        return torch.matmul(scores, key)


class Addition(nn.Module):
    def __init__(self):
        super(Addition, self).__init__()

    def forward(self, x):
        mean = torch.mean(x, dim=1)
        std = torch.std(x, dim=1)
        return torch.cat((mean, std), dim=1)


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, inputs):
        return torch.cat(inputs, dim=1)


class text_model_hc(nn.Module):
    def __init__(self, input_shape):
        super(text_model_hc, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_shape[1], hidden_size=32, batch_first=True, bidirectional=True)
        self.attention = Attention()
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(input_shape[1], 32 * 2)
        self.addition = Addition()
        self.final_dense = nn.Linear(128, 5)

    def forward(self, x):
        x_lstm, _ = self.lstm1(x)
        x_attention = self.attention(x_lstm, x_lstm)
        x_dense = F.relu(self.dense(x))
        x_dense, _ = self.lstm2(x_dense)
        x_add = torch.stack([x_lstm, x_attention, x_dense], dim=0)
        x = torch.sum(x_add, dim=0)
        feat = self.addition(x)
        x = torch.sigmoid(self.final_dense(feat))
        return x, feat


class text_model_nn(nn.Module):
    def __init__(self, input_shape):
        super(text_model_nn, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_shape[1], hidden_size=32, batch_first=True, bidirectional=True)
        self.attention = Attention()
        self.dense1 = nn.Linear(64, 128)
        self.addition = Addition()
        self.dense2 = nn.Linear(128 * 2, 128)
        self.final_dense = nn.Linear(128, 5)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.attention(x, x)
        x = self.dense1(x)
        x = self.addition(x)
        feat = self.dense2(x)
        x = torch.sigmoid(self.final_dense(feat))
        return x, feat


class text_model_b5(nn.Module):
    def __init__(self):
        super(text_model_b5, self).__init__()
        self.dense = nn.Linear(10, 5)

    def forward(self, input_1, input_2):
        X = torch.cat((input_1, input_2), dim=1)
        X = torch.sigmoid(self.dense(X))
        return X

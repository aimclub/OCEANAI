#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Архитектуры аудио моделей для Torch
"""

from __future__ import print_function

import torch.nn as nn
import torchvision.models as models


class audio_model_hc(nn.Module):
    def __init__(self, input_size=25):
        super(audio_model_hc, self).__init__()

        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(64, 128, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(128, 5)

    def extract_features(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        return x[:, -1, :]

    def forward(self, x):
        features = self.extract_features(x)
        x = self.dropout2(features)
        x = self.fc(x)
        return x, features


class audio_model_nn(nn.Module):
    def __init__(self, input_size=512):
        super(audio_model_nn, self).__init__()

        self.vgg = models.vgg16(weights=None)
        self.vgg.classifier = nn.Identity()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 5)

    def extract_features(self, x):
        x = self.vgg.features(x)
        x = self.flatten(x.permute(0, 2, 3, 1))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x

    def forward(self, x):
        features = self.extract_features(x)
        x = self.fc3(features)
        return x, features


class audio_model_b5(nn.Module):
    def __init__(self, input_size=32):
        super(audio_model_b5, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Утилиты модели ResNet50
"""
from __future__ import print_function

import torch
from torchvision import transforms
from PIL import Image


def preprocess_input(fp):
    class PreprocessInput(torch.nn.Module):
        def init(self):
            super(PreprocessInput, self).init()

        def forward(self, x):
            x = x.to(torch.float32)
            x = torch.flip(x, dims=(0,))
            # x[0, :, :] -= 91.4953
            # x[1, :, :] -= 103.8827
            # x[2, :, :] -= 131.0912
            x[0, :, :] -= 93.5940
            x[1, :, :] -= 104.7624
            x[2, :, :] -= 129.1863
            return x

    def get_img_torch(img, target_size=(224, 224)):
        transform = transforms.Compose([transforms.PILToTensor(), PreprocessInput()])
        img = img.resize(target_size, Image.Resampling.NEAREST)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        return img

    return get_img_torch(fp)

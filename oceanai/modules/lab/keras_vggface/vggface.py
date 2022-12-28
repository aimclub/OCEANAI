#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модель VGGFace для Keras
"""

from __future__ import print_function
from oceanai.modules.lab.keras_vggface.models import RESNET50, VGG16, SENET50

def VGGFace(
    include_top = True, model = 'vgg16', weights = 'vggface', input_tensor = None, input_shape = None, pooling = None,
    classes = None
):
    if weights not in {'vggface', None}: raise ValueError

    if model == 'vgg16':
        if classes is None: classes = 2622

        if weights == 'vggface' and include_top and classes != 2622: raise ValueError

        return VGG16(include_top = include_top, input_tensor = input_tensor, input_shape = input_shape,
                     pooling = pooling, weights = weights, classes = classes)

    if model == 'resnet50':
        if classes is None: classes = 8631

        if weights == 'vggface' and include_top and classes != 8631: raise ValueError

        return RESNET50(include_top = include_top, input_tensor = input_tensor, input_shape = input_shape,
                        pooling = pooling, weights = weights, classes = classes)

    if model == 'senet50':
        if classes is None: classes = 8631

        if weights == 'vggface' and include_top and classes != 8631: raise ValueError

        return SENET50(include_top = include_top, input_tensor = input_tensor, input_shape = input_shape,
                       pooling = pooling, weights = weights, classes = classes)
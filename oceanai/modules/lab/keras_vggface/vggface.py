#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модель VGGFace для Keras
"""

from __future__ import print_function
from oceanai.modules.lab.keras_vggface.models import RESNET50


def VGGFace(
    include_top=True, model="vgg16", weights="vggface", input_shape=None, pooling=None, classes=None
):
    if weights not in {"vggface", None}:
        raise ValueError
    
    if classes is None:
        classes = 8631

    if weights == "vggface" and include_top and classes != 8631:
        raise ValueError

    return RESNET50(
        include_top=include_top,
        input_shape=input_shape,
        pooling=pooling,
        weights=weights,
        classes=classes,
    )

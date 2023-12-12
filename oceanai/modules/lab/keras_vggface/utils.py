#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Утилиты модели VGGFace для Keras
"""

import numpy as np
from keras import backend as K

V1_LABELS_PATH = "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v1.npy"
V2_LABELS_PATH = "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v2.npy"

VGG16_WEIGHTS_PATH = "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5"
VGG16_WEIGHTS_PATH_NO_TOP = (
    "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_vgg16.h5"
)

RESNET50_WEIGHTS_PATH = "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_resnet50.h5"
RESNET50_WEIGHTS_PATH_NO_TOP = (
    "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_resnet50.h5"
)

SENET50_WEIGHTS_PATH = "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_senet50.h5"
SENET50_WEIGHTS_PATH_NO_TOP = (
    "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_senet50.h5"
)

VGGFACE_DIR = "models/vggface"


def preprocess_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {"channels_last", "channels_first"}

    if version == 1:
        if data_format == "channels_first":
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == "channels_first":
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp

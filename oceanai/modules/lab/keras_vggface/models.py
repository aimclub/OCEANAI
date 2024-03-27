#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модель VGGFace для Keras
"""

import warnings

from keras import backend as K
from keras import layers
from keras.layers import (
    Flatten,
    Dense,
    Input,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Activation,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    AveragePooling2D,
)
from keras.models import Model
from keras_applications.imagenet_utils import _obtain_input_shape

from oceanai.modules.lab.keras_vggface import utils


def resnet_identity_block(input_tensor, kernel_size, filters, stage, block, bias=False):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    conv1_reduce_name = "conv" + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = "conv" + str(stage) + "_" + str(block) + "_1x1_increase"
    conv3_name = "conv" + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), use_bias=bias, name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "_bn")(x)
    x = Activation("relu")(x)

    x = Conv2D(filters2, kernel_size, use_bias=bias, padding="same", name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "_bn")(x)
    x = Activation("relu")(x)

    x = Conv2D(filters3, (1, 1), use_bias=bias, name=conv1_increase_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "_bn")(x)

    x = layers.add([x, input_tensor])
    x = Activation("relu")(x)
    return x


def resnet_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), bias=False):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    conv1_reduce_name = "conv" + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = "conv" + str(stage) + "_" + str(block) + "_1x1_increase"
    conv1_proj_name = "conv" + str(stage) + "_" + str(block) + "_1x1_proj"
    conv3_name = "conv" + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), strides=strides, use_bias=bias, name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "_bn")(x)
    x = Activation("relu")(x)

    x = Conv2D(filters2, kernel_size, padding="same", use_bias=bias, name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "_bn")(x)
    x = Activation("relu")(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "_bn")(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=bias, name=conv1_proj_name)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=conv1_proj_name + "_bn")(shortcut)

    x = layers.add([x, shortcut])
    x = Activation("relu")(x)
    return x


def RESNET50(include_top=True, weights="vggface", input_shape=None, pooling=None, classes=8631):
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=K.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    img_input = Input(shape=input_shape)

    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(64, (7, 7), use_bias=False, strides=(2, 2), padding="same", name="conv1_7x7_s2")(img_input)
    x = BatchNormalization(axis=bn_axis, name="conv1_7x7_s2_bn")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    x = resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    x = AveragePooling2D((7, 7), name="avg_pool")(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation="softmax", name="classifier")(x)
    else:
        if pooling == "avg":
            x = GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = GlobalMaxPooling2D()(x)

    model = Model(img_input, x, name="vggface_resnet50")

    return model
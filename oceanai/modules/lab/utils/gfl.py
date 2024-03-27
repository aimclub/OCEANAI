#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль внимания
"""

import tensorflow as tf  # Машинное обучение от Google

from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras import regularizers


class GFL(tf.keras.layers.Layer):
    def __init__(self, output_dim, kernel_initializer="glorot_uniform", kernel_regularizer=None, **kwargs):
        self.output_dim = output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        super(GFL, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_HCF1 = self.add_weight(
            name="W_HCF1",
            shape=(int(input_shape[0][1]), self.output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        self.W_DF1 = self.add_weight(
            name="W_DF1",
            shape=(int(input_shape[2][1]), self.output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        self.W_HCF2 = self.add_weight(
            name="W_HCF2",
            shape=(int(input_shape[1][1]), self.output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        self.W_DF2 = self.add_weight(
            name="W_DF2",
            shape=(int(input_shape[3][1]), self.output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        dim_size1 = int(input_shape[0][1]) + int(input_shape[1][1])
        dim_size2 = int(input_shape[2][1]) + int(input_shape[3][1])

        self.W_HCF = self.add_weight(
            name="W_HCF",
            shape=(dim_size1, self.output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        self.W_DF = self.add_weight(
            name="W_DF",
            shape=(dim_size2, self.output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        super(GFL, self).build(input_shape)

    def call(self, inputs):
        HCF1, HCF2, DF1, DF2 = inputs

        h_HCF1 = K.tanh(K.dot(HCF1, self.W_HCF1))
        h_HCF2 = K.tanh(K.dot(HCF2, self.W_HCF2))
        h_DF1 = K.tanh(K.dot(DF1, self.W_DF1))
        h_DF2 = K.tanh(K.dot(DF2, self.W_DF2))

        h_HCF = K.sigmoid(K.dot(K.concatenate([HCF1, HCF2]), self.W_HCF))
        h_DF = K.sigmoid(K.dot(K.concatenate([DF1, DF2]), self.W_DF))

        h = h_HCF * h_HCF1 + (1 - h_HCF) * h_HCF2 + h_DF * h_DF1 + (1 - h_DF) * h_DF2

        return h

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return [input_shape[0][0], self.output_dim]

    def get_config(self):
        config = {
            "output_dim": self.output_dim,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        }
        base_config = super(GFL, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

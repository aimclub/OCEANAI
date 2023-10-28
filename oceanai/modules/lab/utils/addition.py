#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Слой статистических функционалов (средние значения и стандартные отклонения)
"""

import tensorflow as tf  # Машинное обучение от Google

from keras import backend as K


class Addition(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Addition, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[-1]
        super(Addition, self).build(input_shape)

    def call(self, x):
        # return K.sum(x, axis=1)
        m = K.mean(x, axis=1)
        s = K.std(x, axis=1)
        return K.concatenate((m, s), axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

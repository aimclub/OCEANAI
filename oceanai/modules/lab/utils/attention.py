#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль внимания
"""

import tensorflow as tf  # Машинное обучение от Google


class Attention(tf.keras.layers.Layer):
    def __init__(self, use_scale=False, score_mode="dot", **kwargs):
        super().__init__(**kwargs)
        self.use_scale = use_scale
        self.score_mode = score_mode
        if self.score_mode not in ["dot", "concat"]:
            raise ValueError(f"Received: score_mode={score_mode}. Acceptable values " 'are: ["dot", "concat"]')

    def build(self, input_shape):
        if self.use_scale:
            self.scale = self.add_weight(
                name="scale",
                shape=(),
                initializer="ones",
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.scale = None
        if self.score_mode == "concat":
            self.concat_score_weight = self.add_weight(
                name="concat_score_weight",
                shape=(),
                initializer="ones",
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.concat_score_weight = None
        super().build(input_shape)

    def call(self, query, key):
        if self.score_mode == "dot":
            scores = tf.matmul(query, key, transpose_b=True)
            if self.scale is not None:
                scores *= self.scale
        elif self.score_mode == "concat":
            q_reshaped = tf.expand_dims(query, axis=-2)
            k_reshaped = tf.expand_dims(key, axis=-3)
            if self.scale is not None:
                scores = self.concat_score_weight * tf.reduce_sum(
                    tf.tanh(self.scale * (q_reshaped + k_reshaped)), axis=-1
                )
            else:
                scores = self.concat_score_weight * tf.reduce_sum(tf.tanh(q_reshaped + k_reshaped), axis=-1)

        scores = tf.nn.softmax(scores, axis=-1)
        scores = tf.matmul(scores, key)
        return scores

    def get_config(self):
        config = {"use_scale": self.use_scale, "score_mode": self.score_mode}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

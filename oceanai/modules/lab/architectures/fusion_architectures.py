#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Архитектуры моделей слияния для Torch
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class GFL(nn.Module):
    def __init__(self, output_dim, input_shapes):
        super(GFL, self).__init__()
        self.output_dim = output_dim
        self.W_HCF1 = nn.Parameter(torch.Tensor(input_shapes[0], output_dim))
        self.W_DF1 = nn.Parameter(torch.Tensor(input_shapes[2], output_dim))
        self.W_HCF2 = nn.Parameter(torch.Tensor(input_shapes[1], output_dim))
        self.W_DF2 = nn.Parameter(torch.Tensor(input_shapes[3], output_dim))

        init.xavier_uniform_(self.W_HCF1)
        init.xavier_uniform_(self.W_DF1)
        init.xavier_uniform_(self.W_HCF2)
        init.xavier_uniform_(self.W_DF2)

        dim_size1 = input_shapes[0] + input_shapes[1]
        dim_size2 = input_shapes[2] + input_shapes[3]

        self.W_HCF = nn.Parameter(torch.Tensor(dim_size1, output_dim))
        self.W_DF = nn.Parameter(torch.Tensor(dim_size2, output_dim))

        init.xavier_uniform_(self.W_HCF)
        init.xavier_uniform_(self.W_DF)

    def forward(self, inputs):
        HCF1, HCF2, DF1, DF2 = inputs

        h_HCF1 = torch.tanh(torch.matmul(HCF1, self.W_HCF1))
        h_HCF2 = torch.tanh(torch.matmul(HCF2, self.W_HCF2))
        h_DF1 = torch.tanh(torch.matmul(DF1, self.W_DF1))
        h_DF2 = torch.tanh(torch.matmul(DF2, self.W_DF2))

        h_HCF = torch.sigmoid(torch.matmul(torch.cat((HCF1, HCF2), dim=-1), self.W_HCF))
        h_DF = torch.sigmoid(torch.matmul(torch.cat((DF1, DF2), dim=-1), self.W_DF))

        h = h_HCF * h_HCF1 + (1 - h_HCF) * h_HCF2 + h_DF * h_DF1 + (1 - h_DF) * h_DF2

        return h


class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.layer_norm(x)


class avt_model_b5(nn.Module):
    def __init__(self, input_shapes, output_dim=64, hidden_states=50):
        super(avt_model_b5, self).__init__()

        self.ln_hc_t = LayerNormalization(input_shapes[0])
        self.ln_nn_t = LayerNormalization(input_shapes[1])
        self.ln_hc_a = LayerNormalization(input_shapes[2])
        self.ln_nn_a = LayerNormalization(input_shapes[3])
        self.ln_hc_v = LayerNormalization(input_shapes[4])
        self.ln_nn_v = LayerNormalization(input_shapes[5])

        self.gf_ta = GFL(
            output_dim=output_dim, input_shapes=[input_shapes[0], input_shapes[2], input_shapes[1], input_shapes[3]]
        )
        self.gf_tv = GFL(
            output_dim=output_dim, input_shapes=[input_shapes[0], input_shapes[4], input_shapes[1], input_shapes[5]]
        )
        self.gf_av = GFL(
            output_dim=output_dim, input_shapes=[input_shapes[2], input_shapes[4], input_shapes[3], input_shapes[5]]
        )

        self.fc1 = nn.Linear(output_dim * 3, hidden_states)
        self.fc2 = nn.Linear(hidden_states, 5)

    def forward(self, hc_t, nn_t, hc_a, nn_a, hc_v, nn_v):
        hc_t_n = self.ln_hc_t(hc_t)
        nn_t_n = self.ln_nn_t(nn_t)
        hc_a_n = self.ln_hc_a(hc_a)
        nn_a_n = self.ln_nn_a(nn_a)
        hc_v_n = self.ln_hc_v(hc_v)
        nn_v_n = self.ln_nn_v(nn_v)

        gf_ta_out = self.gf_ta([hc_t_n, hc_a_n, nn_t_n, nn_a_n])
        gf_tv_out = self.gf_tv([hc_t_n, hc_v_n, nn_t_n, nn_v_n])
        gf_av_out = self.gf_av([hc_a_n, hc_v_n, nn_a_n, nn_v_n])

        concat_out = torch.cat([gf_ta_out, gf_tv_out, gf_av_out], dim=-1)

        dense_out = F.relu(self.fc1(concat_out))
        output = torch.sigmoid(self.fc2(dense_out))

        return output


class av_model_b5(nn.Module):
    def __init__(self, input_size=64):
        super(av_model_b5, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

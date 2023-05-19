import paddle
import paddle.fluid as fluid
import os
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import io
#import scipy.io as scio
from scipy.io import loadmat
# from sklearn import preprocessing as pp
import numpy as np
from paddle.nn import functional as F
import zipfile
import natsort
from natsort import ns, natsorted
from matplotlib.pyplot import MultipleLocator
from paddle.nn import Linear

import paddle
import math
from math import sqrt
from paddle import nn
# x = paddle.ones([1,1,5])
# print(x.shape)

class SelfAttention(paddle.nn.Layer):
    dim_in: int
    dim_k: int
    dim_v: int
    dim_in = 4626
    dim_k = 4626
    dim_v = 4626


    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(in_features = dim_in, out_features = dim_k)
        self.linear_k = nn.Linear(in_features = dim_in, out_features = dim_k)
        self.linear_v = nn.Linear(in_features = dim_in, out_features = dim_v)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x1):
        batch,n,dim_in = x1.shape
        assert dim_in == self.dim_in
        q = self.linear_q(x1)
        k = self.linear_k(x1)
        v = self.linear_v(x1)
        k1 = paddle.transpose(k,[0,2,1])
        dist = paddle.bmm(q, k1) * self._norm_fact  # batch, n, n
        m = paddle.nn.Softmax()
        dist1 = m(dist)
        att = paddle.bmm(dist1, v)
        return att

dim_in = 4626
class PoswiseFeedForwardNet(paddle.nn.Layer):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features = dim_in, out_features = dim_in),
            nn.ReLU(),
            nn.Linear(in_features = dim_in, out_features = dim_in)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(dim_in)(output + residual)  # [batch_size, seq_len, d_model]
class EncoderLayer(paddle.nn.Layer):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn =SelfAttention(dim_in = 4626,dim_k = 4626, dim_v = 4626)
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs):
        enc_outputs= self.enc_self_attn(enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs+enc_inputs
class DecoderLayer(paddle.nn.Layer):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = SelfAttention(dim_in = 4626,dim_k = 4626, dim_v = 4626)
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs):
        dec_outputs = self.dec_self_attn(dec_inputs)  # 这里的Q,K,V全是Decoder自己的输入
        dec_outputs = self.pos_ffn(dec_inputs)
        return dec_outputs

class Transformer(paddle.nn.Layer):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = EncoderLayer()
        self.decoder = DecoderLayer()

    def forward(self, enc_inputs):
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(enc_outputs)
        return dec_outputs
Notes = 4626
class FC4(paddle.nn.Layer):
    def __init__(self):
        super(FC4, self).__init__()
        # 定义四层全连接神经网络
        self.fc1 = Linear(in_features=Notes, out_features=Notes)
        self.fc2 = Linear(in_features=Notes, out_features=Notes)
        self.fc3 = Linear(in_features=Notes, out_features=Notes)
        self.fc4 = Linear(in_features=Notes, out_features=Notes)


    def forward(self, inputs):
        x1 = self.fc1(inputs)
        x2 = self.fc2(x1)
        x3 = self.fc3(x2)
        out1 = self.fc4(x3)
        return out1
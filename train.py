import paddle
import paddle.fluid as fluid
import os
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import io
from scipy.io import loadmat
# from sklearn import preprocessing as pp
import numpy as np
from paddle.nn import functional as F
import zipfile
import natsort
from natsort import ns, natsorted
from matplotlib.pyplot import MultipleLocator

from model import SelfAttention, Transformer, FC4

Notes = 4626
alldata = np.load("knn_all.npy")
data1 = alldata
print(data1.shape)
model = Transformer()
model1 = FC4()
BATCH_SIZE2 = 8
epochs = 5000
train_losses = []
train_losses1 = []
opt = paddle.optimizer.SGD(learning_rate=0.0005, parameters=model.parameters())
opt1 = paddle.optimizer.SGD(learning_rate=0.0005, parameters=model1.parameters())
for epoch in range(epochs):
    mini_batchs = [data1[k:k+BATCH_SIZE2] for k in range(0,len(data1),BATCH_SIZE2)]
    for data in mini_batchs:
        enc_input1 =data[:,:,0:10]
        enc_input=np.mean(enc_input1, axis=2) + np.max(enc_input1, axis=2)
        true_label = data[:,:,10]
        enc_input = paddle.to_tensor(enc_input,dtype='float32')
        enc_input = paddle.unsqueeze(enc_input, axis=1)
        true_label = paddle.to_tensor(true_label, dtype='float32')
        true_label = paddle.unsqueeze(true_label, axis=0)
        if epoch<2500:
            label_pre = model(enc_input)
            loss =  F.mse_loss(label_pre, true_label) # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
            loss.backward()
            opt.minimize(loss)
            opt.clear_grad()
        else:
            label_pre = model(enc_input)
            energy_pre = model1(label_pre)
            loss = F.mse_loss(label_pre, true_label)
            loss1 = F.mse_loss(energy_pre, enc_input)
            loss2 = loss + loss1
            loss2.backward()
            opt.minimize(loss2)
            opt1.minimize(loss2)
            opt.clear_grad()
            opt1.clear_grad()
    if epoch<2500:
        train_losses.append(loss.numpy()[0])
    else:
        train_losses.append(loss.numpy()[0])
        train_losses1.append(loss1.numpy()[0])
    print(epoch)
    if epoch%1000 ==0:
       fluid.save_dygraph(model.state_dict(), 'FC8_model3')
fluid.save_dygraph(model.state_dict(), 'FC8_model3')
np.save('train_losses',train_losses)
print(train_losses)
np.save('train_losses1',train_losses1)
print(train_losses1)
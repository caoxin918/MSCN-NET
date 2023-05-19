import numpy as np
import paddle
from paddle.nn import functional as F
from paddle import fluid
import warnings

from model import Transformer

warnings.filterwarnings("ignore")
alldata = np.load("knn_all_test .npy")
enc_inputs_tests = alldata
model = Transformer()
model_dict,opti_state_dict=fluid.load_dygraph("FC8_model3")
model.load_dict(model_dict)
model.eval()
test_FC8_losses1 = []
out = []
for data in enc_inputs_tests:
    enc_input_test1 =data[:,0:10]
    print(enc_input_test1.shape)
    enc_input_test = np.mean(enc_input_test1, axis=1) + np.max(enc_input_test1, axis=1)
    true_label_test = data[:,10]
    enc_input_test = paddle.to_tensor(enc_input_test,dtype='float32')
    true_label_test= paddle.to_tensor(true_label_test, dtype='float32')
    enc_input_test = paddle.unsqueeze(enc_input_test, axis=0)
    enc_input_test = paddle.unsqueeze(enc_input_test, axis=0)
    true_label_test = paddle.unsqueeze(true_label_test, axis=0)
    test_outputs = model(enc_input_test)
    out.append(test_outputs.numpy()[0])
    print(test_outputs.numpy()[0])
    loss_pre =  (F.mse_loss(test_outputs, true_label_test))
    print(loss_pre)
num = 0
for pre in out:
    np.savetxt('test'+repr(num)+'.txt',pre)
    num = num +1
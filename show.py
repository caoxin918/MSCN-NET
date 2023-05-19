import numpy as np
from matplotlib import pyplot as plt

train_losses = np.load('train_losses.npy')
#loss2曲线图
#np.random.seed(1000)
print(train_losses)
y1 = train_losses
# print(max(y1))
#print ("y = %s"% y)
x1 = range(len(y1))
# #print ("x=%s"% x)
plt.plot(y1)
# plt.title("train_losses", fontsize=16)
# plt.xlabel('epoch',fontsize=14)
# plt.ylabel('loss',fontsize=14)
# y_major_locator=MultipleLocator(0.001)
# ax=plt.gca()
# ax.yaxis.set_major_locator(y_major_locator)
plt.show()

train_losses = np.load('train_losses1.npy')
#loss2曲线图
#np.random.seed(1000)
y1 = train_losses
# print(max(y1))
#print ("y = %s"% y)
x1 = range(len(y1))
# #print ("x=%s"% x)
plt.plot(y1)
# plt.title("train_losses", fontsize=16)
# plt.xlabel('epoch',fontsize=14)
# plt.ylabel('loss',fontsize=14)
# y_major_locator=MultipleLocator(0.001)
# ax=plt.gca()
# ax.yaxis.set_major_locator(y_major_locator)
plt.show()
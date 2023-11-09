import numpy as np
import pandas as pd
from MLP import MLP as MP
path = '/home/mahdi/Desktop/CI_lab/HW3.V3/iris.cvs'
# path = 'iris.cvs'
dataset = pd.read_csv(path)
###############################
x = dataset.iloc[:, :4].values.T          # read data
data_length, feature_size = x.shape         # save the shape of data 

########### LABEL #############
label = dataset.iloc[:, 4].values
label = np.where(label == 'Iris-setosa', 1, label).reshape([-1, 1])
label = np.where(label == 'Iris-versicolor', 0, label).reshape([-1, 1])
label = np.where(label == 'Iris-virginica', -1, label).reshape([-1, 1])
y = label.T.reshape(149)
ones = np.array([[1], [0], [0]])
zeros = np.array([[0], [1], [0]])
neg_ones = np.array([[0], [0], [1]])
y = np.where(y == 1, ones, np.where(y == 0, zeros, neg_ones))
train_index = np.random.randint(0, 149, size= 120)
model = MP(4, 3, 7, 7, 3)
# msg = np.round(model.forward(x), 2)
# print(msg[:, :10:])
model.fit(x[:, train_index], y[:, train_index], 0.01, epoches= 10000)
msg = np.round(model.forward(x[:, train_index]), 2)
disp = np.concatenate((train_index.reshape(1, -1), msg), axis=0)
print(np.round(disp[:,:12:], 2))

    

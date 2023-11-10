import numpy as np
np.set_printoptions(suppress=True) # Dont Show as scientific number
import pandas as pd
from MLP import MLP as MP
from datetime import datetime as dt
#____________________________________________________

# Read Data
# path = '/home/mahdi/Desktop/CI_lab/HW3.V3/iris.cvs'
path = 'iris.cvs'
dataset = pd.read_csv(path)
#____________________________________________________

# Initialize Data
x = dataset.iloc[:, :4].values.T        # read data
feature_size, data_length = x.shape     # save the shape of data 
#____________________________________________________

# LABEL
# CVT String to NO.
label = dataset.iloc[:, 4].values
label = np.where(label == 'Iris-setosa',     1, label).reshape([-1, 1])
label = np.where(label == 'Iris-versicolor', 0, label).reshape([-1, 1])
label = np.where(label == 'Iris-virginica', -1, label).reshape([-1, 1])
# Save as y: ndarray
'''
Iris-setosa Iris-versicolor Iris-virginica
    1           0               0
    0           1               0
    0           0               1
'''
y = label.T.reshape(149)
ones = np.array([[1], [0], [0]])
zeros = np.array([[0], [1], [0]])
neg_ones = np.array([[0], [0], [1]])
y = np.where(y == 1, ones, np.where(y == 0, zeros, neg_ones))
#____________________________________________________
# Initialize Train and Test data
all_index = np.arange(data_length)
train120 = np.random.choice(range(data_length), 
                            size= 120, replace= False)
test120 = np.setdiff1d(all_index, train120)

train90 = np.random.choice(range(data_length), 
                            size= 90, replace= False)
test90 = np.setdiff1d(all_index, train90)
#____________________________________________________
# Make a model
# Example
# model = MP('4', 4, 3, 7, 7, 3)
# model.fit(x[:, train120], y[:, train120], 0.01, epoches= 10000)
# msg = np.round(model.forward(data_test), 2)
# print(train120[:12:])

model2_3 = MP('2L 3H', 4, 3, 3)
model2_5 = MP('2L 5H', 4, 3, 5)
model3_53 = MP('3L 5-3 H',4, 3, 5, 3)
model3_77 = MP('3L 7-7 H', 4, 3, 7, 7)
model4_533 = MP('4L 5-3-3 H', 4, 3, 5, 3, 3)
model4_773 = MP('4L 7-7-3 H', 4, 3, 7, 7, 3)
model5_3553 = MP('5L 3-5-5-3 H', 4, 3, 3, 5, 5, 3)

model = [model2_3, model2_5,
         model3_53, model3_77,
         model4_533, model4_773,
         model5_3553]
# Writing on file
file_out = open('OUTPUT.txt', 'w')
today = dt.now()
file_out.write(f"{today}\n")
# Train on 90 / 150 and 100 epoches
for obj in model:
    data_train = x[:, train90]
    data_test  = x[:, test90]
    label_train = y[:, train90]
    label_test = y[:, test90]
    obj.fit(x= data_train,
              y= label_train,
              learning_rate= 0.01,
              epoches= 100
               )
    test_out = np.round(obj.forward(data_test), 2)
    title = f"{obj.name},\t 100 epoches, 90/150\n"
    msg_out = np.concatenate((test90[::5].reshape([1, -1]), test_out[:, ::5]), axis= 0)
    msg_out = np.array2string(msg_out)
    file_out.write(f"{title}\n{msg_out}\n")
# Train on 90 / 150 and 500 epoches
for obj in model:
    data_train = x[:, train90]
    data_test  = x[:, test90]
    label_train = y[:, train90]
    label_test = y[:, test90]
    obj.fit(x= data_train,
              y= label_train,
              learning_rate= 0.01,
              epoches= 500
              )
    test_out = np.round(obj.forward(data_test), 2)
    title = f"{obj.name},\t 500 epoches, 90/150\n"
    msg_out = np.concatenate((test90[::5].reshape([1, -1]), test_out[:, ::5]), axis= 0)
    msg_out = np.array2string(msg_out)
    file_out.write(f"{title}\n{msg_out}\n")
# Train on 120 / 150 and 100 epoches
for obj in model:
    data_train = x[:, train120]
    data_test  = x[:, test120]
    label_train = y[:, train120]
    label_test = y[:, test120]
    obj.fit(x= data_train,
              y= label_train,
              learning_rate= 0.01,
              epoches= 100
              )
    test_out = np.round(obj.forward(data_test), 2)
    title = f"{obj.name},\t 100 epoches, 120/150\n"
    msg_out = np.concatenate((test120[::5].reshape([1, -1]), test_out[:, ::5]), axis= 0)
    msg_out = np.array2string(msg_out)
    file_out.write(f"{title}\n{msg_out}\n")
# Train on 120 / 150 and 500 epoches
for obj in model:
    data_train = x[:, train120]
    data_test  = x[:, test120]
    label_train = y[:, train120]
    label_test = y[:, test120]
    obj.fit(x= data_train,
              y= label_train,
              learning_rate= 0.01,
              epoches= 500
              )
    test_out = np.round(obj.forward(data_test), 2)
    title = f"{obj.name},\t 500 epoches, 120/150\n"
    msg_out = np.concatenate((test120[::5].reshape([1, -1]), test_out[:, ::5]), axis= 0)
    msg_out = np.array2string(msg_out)
    file_out.write(f"{title}\n{msg_out}\n")
    

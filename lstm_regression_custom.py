import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

def read_VA2np(filepath):
    with open(filepath, "r") as f:
        va_list = []
        for line in f:
            mc = line.split(' ')
            va_list.append(mc[:-1])
            
    return np.array(va_list).astype(np.float64).T

def read_GT2np(filepath):
    with open(filepath, "r") as f:
        va_list = []
        for line in f:
            mc = line.split(' ')
            va_list.append(mc)
            
    return np.array(va_list).astype(np.float64).T

def read_index2lst(indexpathlist):
    
    idxlist = []
    for k in range(len(indexpathlist)):
        with open(indexpathlist[k], "r") as f:
            tmplist = []
            for line in f:
                
                tmplist.append(line[:-1])
                
            idxlist.append(tmplist)
                    
    return idxlist

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length-1]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

va_np = read_VA2np('./flim_data/Avengers_EG_v2_300_6dof.xtt')
gt_np = read_GT2np('./flim_data/Avengers_EG_v2.xtt')
inputidxlist = ['./flim_data/Avengers_EG_v2_roll.txt', './flim_data/Avengers_EG_v2_pitch.txt', './flim_data/Avengers_EG_v2_heave.txt']
idxlist = read_index2lst(inputidxlist)

print(va_np)
print(gt_np)
print(idxlist)

seq_length = 4

# x_pitch, x_ = sliding_windows(va_np[0], seq_length)
# x_yaw, x_ = sliding_windows(va_np[1], seq_length)
x, y = sliding_windows(va_np[2], seq_length)
# x_sway, x_ = sliding_windows(va_np[3], seq_length)
# x_sway, x_ = sliding_windows(va_np[4], seq_length)
# x_sway, x_ = sliding_windows(va_np[5], seq_length)

#x = np.vstack([x_yaw, x_roll, x_sway])

#y, y_ = sliding_windows(gt_np[0], seq_length)
print('len(va_np[2])(roll): %d'%(len(va_np[2])))
print('len(x_roll): %d'%(len(x)))
print('len(y): %d'%(len(y)))
print(x[0])
print(x.shape)
print(y[0])
print(y.shape)

train_size = int(len(y) * 0.67)
test_size = len(y) - train_size
print('train_size: %d\n' %(train_size))
print('test_size: %d\n' %(test_size))

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

# training_set = pd.read_csv('airline-passengers.csv')
# #training_set = pd.read_csv('shampoo.csv')

# training_set = training_set.iloc[:,1:2].values

# plt.plot(training_set, label = 'Shampoo Sales Data')
# #plt.show()

# '''
# Dataloading
# '''
# def sliding_windows(data, seq_length):
#     x = []
#     y = []

#     for i in range(len(data)-seq_length-1):
#         _x = data[i:(i+seq_length)]
#         _y = data[i+seq_length]
#         x.append(_x)
#         y.append(_y)

#     return np.array(x),np.array(y)

# sc = MinMaxScaler()
# training_data = sc.fit_transform(training_set)
# print(training_data[0:10])
# print('len(training_data): %d' %(len(training_data)))

# seq_length = 4
# x, y = sliding_windows(training_data, seq_length)
# #print(x[0])
# #print(y[0])
# # print('len(x): %d' %(len(x)))
# # print('len(y): %d' %(len(y)))

# train_size = int(len(y) * 0.67)
# test_size = len(y) - train_size
# print('train_size: %d\n' %(train_size))
# print('test_size: %d\n' %(test_size))

# dataX = Variable(torch.Tensor(np.array(x)))
# dataY = Variable(torch.Tensor(np.array(y)))

# trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
# trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

# testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
# testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

'''
LSTM Model
'''
class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out


'''
Training
'''
num_epochs = 2000
learning_rate = 0.01

input_size = 1
hidden_size = 2
num_layers = 1

num_classes = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = criterion(outputs, trainY)
    
    loss.backward()
    
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# lstm.eval()
# train_predict = lstm(dataX)

# data_predict = train_predict.data.numpy()
# dataY_plot = dataY.data.numpy()

# data_predict = sc.inverse_transform(data_predict)
# dataY_plot = sc.inverse_transform(dataY_plot)

# plt.axvline(x=train_size, c='r', linestyle='--')

# plt.plot(dataY_plot)
# plt.plot(data_predict)
# plt.suptitle('Time-Series Prediction')
# plt.show()
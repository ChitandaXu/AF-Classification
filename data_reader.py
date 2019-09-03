import numpy as np
from pandas import DataFrame
import scipy.io as sio
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

os.chdir('E:/xuxuexiang/challenge 2017/master')

# 获取文件名列表
frecord = open('training2017/RECORDS', 'r')
lines = frecord.readlines()
for i in range(0, len(lines)):
    lines[i] = lines[i].replace('\n','')

# 规定输入大小, 计算样本大小
input_size = 10100
number_of_sample = len(lines)

# 填充样本
X = np.zeros((number_of_sample, input_size))
segment = np.empty(number_of_sample, dtype = int)
for i in range(0, number_of_sample):
    filename = lines[i]
    dictionary = sio.loadmat('training2017/' + filename)
    data = dictionary['val'][0,:]
    if (input_size - len(data)) <= 0:
        X[i, :] = data[0:input_size]
    else:
        diff = data[0:(input_size - len(data))]
        goal = np.hstack((data, diff))
        while len(goal) != input_size:
            diff = data[0: (input_size - len(goal))]
            goal = np.hstack((goal, diff))
        X[i, :] = goal
 
pd_data = DataFrame(X)
pd_data.to_csv('train_data.csv', index=False)  

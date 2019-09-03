# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:07:00 2019

@author: Administrator
"""
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split

def prepare_data():
    print("Data loading...")
    data = pd.read_csv('train_data.csv')
    label = pd.read_csv('training2017/REFERENCE.csv',header=None)
    print("Data loading finished.")
    size = data.shape[0]
    
    target = np.zeros((size, 1))
    for i in range(size):
        if label[1].loc[i] == 'N':
            target[i] = 0
        elif label[1].loc[i] == 'A':
            target[i] = 1
        elif label[1].loc[i] == 'O':
            target[i] = 2
        elif label[1].loc[i] == '~':
            target[i] = 3
    
    
    np.random.seed(7)
    
    #转成one-hot matrix
#    Label_set = np.zeros((size, nb_classes))
#    for i in range(size):
#        dummy = np.zeros((nb_classes))
#        dummy[int(target[i])] = 1
#        Label_set[i, :] = dummy
    
    data = (data - data.mean())/(data.std()) #Some normalization here
    data = np.expand_dims(data, axis=2) #For Keras's data input size
    
    values = [i for i in range(size)]
    permutations = np.random.permutation(values)
    data = data[permutations, :]
#    Label_set = Label_set[permutations, :]
    target = target[permutations, :]
    
    X, X_test, y, y_test = train_test_split(
            data, target, test_size=0.3, stratify=target)
    return (X, X_test, y, y_test)

def prepare_data_onehot():
    print("Data loading...")
    data = pd.read_csv('train_data.csv')
    label = pd.read_csv('training2017/REFERENCE.csv',header=None)
    print("Data loading finished.")
    size = data.shape[0]
    
    target = np.zeros((size, 1))
    for i in range(size):
        if label[1].loc[i] == 'N':
            target[i] = 0
        elif label[1].loc[i] == 'A':
            target[i] = 1
        elif label[1].loc[i] == 'O':
            target[i] = 2
        elif label[1].loc[i] == '~':
            target[i] = 3
    
    
    np.random.seed(7)
    nb_classes = 4
    
    #转成one-hot matrix
    Label_set = np.zeros((size, nb_classes))
    for i in range(size):
        dummy = np.zeros((nb_classes))
        dummy[int(target[i])] = 1
        Label_set[i, :] = dummy
    
    data = (data - data.mean())/(data.std()) #Some normalization here
    data = np.expand_dims(data, axis=2) #For Keras's data input size
    
    values = [i for i in range(size)]
    permutations = np.random.permutation(values)
    data = data[permutations, :]
    Label_set = Label_set[permutations, :]
    target = target[permutations, :]
    
    X, X_test, y, y_test = train_test_split(
            data, Label_set, test_size=0.3, stratify=Label_set)
    return (X, X_test, y, y_test)
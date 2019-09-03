# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 19:06:13 2018

@author: Administrator
"""


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D

number_of_classes = 4
input_length = 10100
channels = 1

def create_model():
    model = Sequential()
    model.add(Conv1D(128, 55, activation='relu', input_shape=(input_length, channels)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 25, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 10, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(2))

    
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


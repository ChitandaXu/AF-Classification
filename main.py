# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 19:06:13 2018

@author: Administrator
"""
import os
os.chdir('E:/xuxuexiang/challenge 2017/master')

from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import resnet
from model import create_model
from loss_history import LossHistory
from prepare_data import prepare_data, prepare_data_onehot


X, X_test, y, y_test = prepare_data_onehot()
# print('Total training size is ', size)
input_length = 10100
channels = 1
# model = create_model()
nb_classes = 4
model = resnet.ResnetBuilder.build_resnet_34((channels, input_length), nb_classes)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = LossHistory()

checkpointer = ModelCheckpoint(filepath='Conv_models/Best_model.h5', monitor='val_acc', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
hist = model.fit(X, y, batch_size=30, epochs=200, verbose=2, callbacks=[checkpointer, history, early_stopping], validation_split=0.1, shuffle=True)

model = load_model('Conv_models/Best_model.h5')

def change(x):  #From boolean arrays to decimal arrays
    answer = np.zeros((np.shape(x)[0]))
    for i in range(np.shape(x)[0]):
        max_value = max(x[i, :])
        max_index = list(x[i, :]).index(max_value)
        answer[i] = max_index
    return answer.astype(np.int)

pred_test = model.predict(X_test)
pred_overall = model.predict(X)
test_score = accuracy_score(change(y_test), change(pred_test))
overall_score = accuracy_score(change(y), change(pred_overall))
print('Test score is ', test_score)
print('Overall score is ', overall_score)

history.loss_plot('epoch')

pd.DataFrame(confusion_matrix(change(y), change(pred_overall))).to_csv(path_or_buf='Conv_models/Result_Conf' + str(format(overall_score, '.4f')) + '.csv', index=None, header=None)
pd.DataFrame(confusion_matrix(change(y_test), change(pred_test))).to_csv(path_or_buf='Conv_models/Result_Conf' + str(format(test_score, '.4f')) + '.csv', index=None, header=None)

X, X_test, y, y_test = prepare_data()
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('flatten_2').output)

featureX = intermediate_layer_model.predict(X)
featureX = pd.DataFrame(featureX)

gbr = GradientBoostingClassifier(n_estimators=300, max_depth=2, min_samples_split=2, learning_rate=0.1)
gbr.fit(featureX, y.ravel())

# compute test accuracy
featureX_test = intermediate_layer_model.predict(X_test)
y_test_pred = gbr.predict(featureX_test)
test_score = accuracy_score(y_test, y_test_pred)

# compute overall accuracy
featureX = intermediate_layer_model.predict(X)
y_pred = gbr.predict(featureX)
overall_score = accuracy_score(y, y_pred)

print(test_score)
print(overall_score)
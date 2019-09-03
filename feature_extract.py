# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:07:55 2019

@author: Administrator
"""
import os
import pandas as pd
from keras.models import Model,load_model
from prepare_data import prepare_data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

#Preparing Indermediate model

os.chdir('E:/xuxuexiang/challenge 2017/master')

X, X_test, y, y_test = prepare_data()
model = load_model('Conv_models/Best_model.h5')
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('flatten_1').output)
intermediate_layer_model.summary()

#predict to get featured data

featureX = intermediate_layer_model.predict(X)
featureX = pd.DataFrame(featureX)
print('feature X shape:', featureX.shape)
featureX.head(5)  #The features are unnamed now

gbr = GradientBoostingClassifier(n_estimators=300, max_depth=2, min_samples_split=2, learning_rate=0.1)
gbr.fit(featureX, y.ravel())

# compute test accuracy
featureX_test = intermediate_layer_model.predict(X_test)
y_test_pred = gbr.predict(featureX_test)
metrics.accuracy_score(y_test, y_test_pred)

# compute overall accuracy
featureX = intermediate_layer_model.predict(X)
y_pred = gbr.predict(featureX)
metrics.accuracy_score(y, y_pred)
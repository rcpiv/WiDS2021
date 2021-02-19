# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 21:56:03 2021

@author: Robby
"""

#%% Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Load in Data
import os
os.chdir(r"C:\Users\Robby\Desktop\IAA\Personal Projects\WiDS")

train = pd.read_csv(r'train_split.csv')
valid = pd.read_csv(r'valid.csv')

train2 = pd.read_csv(r'train_imput_split.csv')
valid2 =  pd.read_csv(r'valid_imput.csv')

#%% Split off target column
train_y = train.diabetes_mellitus
train_x = train.drop('diabetes_mellitus', axis=1)

train2_y = train2.diabetes_mellitus
train2_x = train2.drop('diabetes_mellitus', axis=1)

valid_y = valid.diabetes_mellitus
valid_x = valid.drop('diabetes_mellitus', axis=1)

valid2_y = valid2.diabetes_mellitus
valid2_x = valid2.drop('diabetes_mellitus', axis=1)

#%% Get dummies
train_x = pd.get_dummies(train_x)
train2_x = pd.get_dummies(train2_x)
valid_x = pd.get_dummies(valid_x)
valid2_x = pd.get_dummies(valid2_x)


#%% Downsample
from sklearn.model_selection import train_test_split as tts
train_x_ds, a, train_y_ds, b = tts(train_x,train_y, test_size = .9, random_state= 1234)

#%% Set Up
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
#%% Set Params
params = {
    'learning_rate' : .3,
    'boosting_type' : 'dart',
    'objective' : 'binary',
    'metric' : 'auc',
    'sub_feature' : .8,
    'num_leaves' : 50,
    'min_data' : 20,
    'max_depth' : 10,
    'feature_pre_filter': False,
    'verbose': -1
    }

d_train = lgb.Dataset(train_x_ds, label=train_y_ds)
#%% LGBM
lgbm = lgb.cv(params, d_train, num_boost_round= 100, nfold=5, seed = 42)
print(np.max(lgbm['auc-mean'])) # 0.82

#%% Parameter Tuning
gridsearch_params = [
    (min_data, max_depth)
    for min_data in range(52,55,1)
    for max_depth in range (4,5,1)]
        
max_AUC = float(0)
best_params = None
for min_data, max_depth in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_data))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_data'] = min_data
    # Run CV
    cv_results = lgb.cv(
        params,
        d_train,
        num_boost_round=100,
        seed=42,
        nfold=5,
        metrics={'auc'},
        #early_stopping_rounds=10
    )
    # Update best MAE
    mean_auc = np.max(cv_results['auc-mean'])
    print("\tAUC {}".format(mean_auc))
    if mean_auc > max_AUC:
        max_AUC = mean_auc
        best_params = (max_depth,min_data)
print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_AUC))

#%% Set Params
params['max_depth'] = 4
params['min_data'] = 53

#%% Parameter Tuning
max_AUC = float(0)
best_params = None
for num_leaves in range(12,32,1):
    print("CV with num_leaves={}".format(num_leaves))
    # Update our parameters
    params['num_leaves'] = num_leaves
    # Run CV
    cv_results = lgb.cv(
        params,
        d_train,
        num_boost_round=100,
        seed=42,
        nfold=5,
        metrics={'auc'},
        #early_stopping_rounds=10
    )
    # Update best MAE
    mean_auc = np.max(cv_results['auc-mean'])
    print("\tAUC {}".format(mean_auc))
    if mean_auc > max_AUC:
        max_AUC = mean_auc
        best_params = (num_leaves)
print("Best params: {}, AUC: {}".format(best_params, max_AUC))

#%% Parameter Tuning
max_AUC = float(0)
best_params = None
for colsample in range(41,62,1):
    print("CV with colsample={}".format(colsample/100))
    # Update our parameters
    params['sub_feature'] = colsample/100
    # Run CV
    cv_results = lgb.cv(
        params,
        d_train,
        num_boost_round=100,
        seed=42,
        nfold=5,
        metrics={'auc'},
        #early_stopping_rounds=10
    )
    # Update best MAE
    mean_auc = np.max(cv_results['auc-mean'])
    print("\tAUC {}".format(mean_auc))
    if mean_auc > max_AUC:
        max_AUC = mean_auc
        best_params = (colsample)
print("Best params: {}, AUC: {}".format(best_params, max_AUC))

#%% Set Param
params['num_leaves'] = 16
params['sub_feature'] = .52
#%% Parameter Tuning
max_AUC = float(0)
best_params = None
for eta in range(200,241,10):
    print("CV with eta={}".format(eta/1000))
    # Update our parameters
    params['learning_rate'] = eta/1000
    # Run CV
    cv_results = lgb.cv(
        params,
        d_train,
        num_boost_round=100,
        seed=42,
        nfold=5,
        metrics={'auc'},
        #early_stopping_rounds=10
    )
    # Update best MAE
    mean_auc = np.max(cv_results['auc-mean'])
    print("\tAUC {}".format(mean_auc))
    if mean_auc > max_AUC:
        max_AUC = mean_auc
        best_params = (eta/1000)
print("Best params: {}, AUC: {}".format(best_params, max_AUC))

#%% Set Params
params['learning_rate'] = .22

#%% Check Dart
params['boosting_type'] = 'dart'
cv_results = lgb.cv(
       params,
       d_train,
       num_boost_round = 250,
       seed=42,
       nfold=5,
       metrics='auc'
       )

# Update best MAE
mean_auc = np.max(cv_results['auc-mean'])
print("\tAUC {}".format(mean_auc))

#%% Best Model
model = lgb.train(
    params,
    d_train,
    num_boost_round = 250,
    )

preds = model.predict(train_x)
print(roc_auc_score(train_y, preds))
# 0.84
#%% Fit to whole data
d_train2 = lgb.Dataset(train_x,label= train_y)
d_valid = lgb.Dataset(valid_x, label=valid_y)

model2 = lgb.train(params,
                   d_train2,
                   num_boost_round= 250)

preds2= model2.predict(train_x)
print(roc_auc_score(train_y, preds2))
# 0.866
#%% Validate 
predsv = model.predict(valid_x)
print(roc_auc_score(valid_y,predsv))
#0.838
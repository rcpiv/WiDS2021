# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 15:15:25 2021

@author: Robby
"""

#%% Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
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

#%% XGBoost
import xgboost as xgb

dtrain = xgb.DMatrix(train_x,label=train_y)
dtest = xgb.DMatrix(valid_x,label=valid_y)

params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'binary:logistic',
}

params['eval_metric'] ='auc'
num_boost_rounds = 250

model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_rounds,
    evals=[(dtest,'Test')],
    early_stopping_rounds=10
    )

print("Best AUC: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))

cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_rounds,
    seed=1234,
    nfold=5,
    metrics={'auc'},
    early_stopping_rounds=10
)
cv_results
cv_results['test-auc-mean'].max()
#%% Grid for max depth and child weight
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(3,15,3)
    for min_child_weight in range(1,10,2)]
        
# Define initial best params and AUC
max_AUC = float(0)
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_rounds,
        seed=42,
        nfold=5,
        metrics={'auc'},
        early_stopping_rounds=10
    )
    # Update best MAE
    mean_auc = cv_results['test-auc-mean'].max()
    boost_rounds = cv_results['test-auc-mean'].argmax()
    print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
    if mean_auc > max_AUC:
        max_AUC = mean_auc
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_AUC))

#%% 3 and 3
params['max_depth'] = 3
params['min_child_weight'] = 3

#%% subsample and colsample
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(6,11)]
    for colsample in [i/10. for i in range(6,11)]
]

max_auc = float(0)
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_rounds,
        seed=42,
        nfold=5,
        metrics={'auc'},
        early_stopping_rounds=10
    )
    # Update best score
    mean_auc = cv_results['test-auc-mean'].max()
    boost_rounds = cv_results['test-auc-mean'].argmax()
    print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
    if mean_auc > max_auc:
        max_auc = mean_auc
        best_params = (subsample,colsample)
print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_auc))

#%% Assign 1&1
params['subsample'] = 1
params['colsample'] = 1

#%% ETA
# This can take some time…
max_auc = float(0)
best_params = None
for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
    # We update our parameters
    params['eta'] = eta
    # Run and time CV
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_rounds,
            seed=1234,
            nfold=5,
            metrics=['auc'],
            early_stopping_rounds=10
          )
    # Update best score
    mean_auc = cv_results['test-auc-mean'].max()
    boost_rounds = cv_results['test-auc-mean'].argmax()
    print("\tAUC {} for {} rounds\n".format(mean_auc, boost_rounds))
    if mean_auc > max_auc:
        max_auc = mean_auc
        best_params = eta
print("Best params: {}, AUC: {}".format(best_params, max_auc))

#%% ETA
params['eta'] = .2

#%% Test
model_xgb = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_rounds,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10)

print("Best AUC: {:.2f} in {} rounds".format(model_xgb.best_score, model_xgb.best_iteration+1))

#%% Best Model
num_boost_round = model_xgb.best_iteration + 1
model_xgb = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")]
)

preds_xgb = model_xgb.predict(dtest)
print(roc_auc_score(valid_y, preds_xgb))
# 0.8525

#%% Test Set
test = pd.read_csv(r"C:\Users\Robby\Desktop\IAA\Personal Projects\WiDS\unlabeled.csv").drop('Unnamed: 0', axis = 1)
test2 = pd.read_csv(r"C:\Users\Robby\Desktop\IAA\Personal Projects\WiDS\unlabeled2.csv").drop('Unnamed: 0', axis = 1)

eid = test.encounter_id
test = test.drop('encounter_id', axis=1)

test = pd.get_dummies(test)
test2 = pd.get_dummies(test2)

dtest2 = xgb.DMatrix(test)

preds_t = model_xgb.predict(dtest2)

pred_dict = {"encounter_id":eid, "diabetes_mellitus": preds_t}

pred_df = pd.DataFrame.from_dict(pred_dict)
pred_df.to_csv(r'XGB_preds.csv', index=False)


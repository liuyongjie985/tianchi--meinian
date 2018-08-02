# -*- coding: UTF-8 -*-
import types
import json
import sys
import csv
import chardet
import time
import datetime
import math
import os
import re
import random
import operator

import lightgbm as lgb
import xgboost as xgb

import numpy as np
import pandas as pd

# Limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV



def my_score_func(y, y_pred):
    y = np.exp(y)
    y_pred = np.exp(y_pred)
    y_pred = y_pred.reshape(y.shape)
    y = y
    _score = np.mean(np.square(np.log2(y_pred + 1) - np.log2(y + 1)))
    return _score
my_scorer = make_scorer(score_func=my_score_func)



def do_model(pred_type=None):
    isServer = 0
    if pred_type == 0:
        real_pred_type = '收缩压'
    elif pred_type == 1:
        real_pred_type = '舒张压'
    elif pred_type == 2:
        real_pred_type = '血清甘油三酯'
    elif pred_type == 3:
        real_pred_type = '血清高密度脂蛋白'
    else:
        real_pred_type = '血清低密度脂蛋白'

    # load 5 files
    train_selection = pd.read_csv('../data/NumericData_clean_train.csv')
    test_selection = pd.read_csv('../data/NumericData_clean_test.csv')
    train_selection_category = pd.read_csv('../data/CategoryData_clean_train.csv')
    test_selection_category = pd.read_csv('../data/CategoryData_clean_test.csv')
    y_train = pd.read_csv('../data/y_train_clear.csv')

    # load valid cols
    train_selection = train_selection.ix[:, 1:]
    test_selection = test_selection.ix[:, 1:]
    train_selection_category = train_selection_category.ix[:, 2:]
    test_selection_category = test_selection_category.ix[:, 2:]

    # merge all types of feature
    train_selection_all = pd.concat([train_selection, train_selection_category], axis=1)
    test_selection_all = pd.concat([test_selection, test_selection_category], axis=1)

    # get x_train, y_train and test
    train_selection_all = train_selection_all.fillna(train_selection.mean())
    x_train = train_selection_all.values

    y_train = y_train.ix[:, [real_pred_type]]
    y_train = y_train.fillna(y_train.mean())
    y_train = y_train.values
    y_train = np.log(y_train)

    test_selection_all = test_selection_all.fillna(test_selection.mean())
    test = test_selection_all.values

    # start modeling
    train_lightGBM_origin(x_train, y_train, test, real_pred_type)



def train_lightGBM_origin(x_train, y_train, test, pred_type=None):
    # x_test = x_train[20000:30000, :]
    # y_test = y_train[20000:30000, :]
    # x_test = x_train[10000:20000, :]
    # y_test = y_train[10000:20000, :]
    x_test = x_train[:10000, :]
    y_test = y_train[:10000, :]
    # x_train = x_train[2000:, :]
    # y_train = y_train[2000:, :]

    # shuffle data
    # shuffle_index = np.random.permutation(np.arange(len(y_train)))
    # x_train = x_train[shuffle_index]
    # y_train = y_train[shuffle_index]
    # rate = 0.7
    #
    # train_len = int(len(x_train) * rate)
    # x_test = x_train[train_len:]
    # y_test = y_train[train_len:]
    # x_train = x_train[:train_len]
    # y_train = y_train[:train_len]

    # train
    print 'Start train...'

    lgb_params_final = lgb.LGBMRegressor(task='train', boosting_type='gbdt', objective='regression', num_leaves=110,
                                         learning_rate=0.005, n_estimators=3500, min_data_in_leaf=20, max_depth=12,
                                         subsample=0.8, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5,
                                         seed=0, verbose=1)
    # lgb_params_final = lgb.LGBMRegressor()
    model_lgb_final = lgb_params_final.fit(x_train, y_train)

    # predict
    y_pred_final = model_lgb_final.predict(x_test, num_iteration=model_lgb_final.best_iteration_)
    y_pred_final_test = model_lgb_final.predict(test, num_iteration=model_lgb_final.best_iteration_)
    y_pred_final.resize(y_pred_final.shape[0], 1)
    y_pred_final_test.resize(y_pred_final_test.shape[0], 1)
    print str(pred_type) + ' predict over'
    print 'The score of prediction is:', my_score_func(y_test, y_pred_final)
    y_pred_final_test = np.exp(y_pred_final_test)

    # write result to txt
    if pred_type == '收缩压':
        pred_file = open('../data/pred_1', 'w')
    elif pred_type == '舒张压':
        pred_file = open('../data/pred_2', 'w')
    elif pred_type == '血清甘油三酯':
        pred_file = open('../data/pred_3', 'w')
    elif pred_type == '血清高密度脂蛋白':
        pred_file = open('../data/pred_4', 'w')
    else:
        pred_file = open('../data/pred_5', 'w')
    for line in y_pred_final_test:
        pred_file.write(str(line[0]) + '\n')
    pred_file.close()



def mergeResult():
    vids = pd.read_csv('../data/meinian_round1_test_b_20180505.csv')
    vids = vids.ix[:, :1].values

    for i in range(5):
        fileName = 'pred_' + str(i + 1)
        pred_list = []
        pred_file = open('../data/' + fileName, 'r')
        lines = pred_file.readlines()
        for line in lines:
            pred_list.append(line.replace('\n', ''))
        pred_list = np.array(pred_list)
        pred_list.resize(pred_list.shape[0], 1)
        vids = np.concatenate((vids, pred_list), axis=1)
        pred_file.close()

    result = pd.DataFrame(vids)
    result.to_csv('../data/submit_20180506_000000_temp.csv', index=False, index_label=False, header=None)

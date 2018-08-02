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

import numpy as np
import pandas as pd

# Limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))



def make_wash_train():
    # wash train_set.csv
    train = pd.read_csv('../data/train_set.csv')
    log = open('../data/NumericData_tableID', 'r')
    # log1 = open('../data/categoricalData_tableID', 'r')
    # log2 = open('../data/StringData_tableID', 'r')
    lines = log.readlines()
    table_id_list = []
    for line in lines:
        table_id_list.append(str(line.replace('\n', '').replace('\r', '')))
    train_selection = train.ix[:, table_id_list]
    # change str to float
    train_selection = train_selection.convert_objects(convert_numeric=True)

    # solve the not-float data
    for table in table_id_list:
        print 'checking table : ' + str(table) + '...'
        not_float_index = []
        one_col = train_selection.ix[:, table]
        col_len = len(one_col)
        for index in range(col_len):
            # print index
            try:
                try:
                    number = float(one_col[index + 0])
                    if one_col.dtype != 'float64' and one_col.dtype != 'float32' and one_col.dtype != 'float':
                        one_col[index] = number
                except Exception, e:
                    num = filter(str.isdigit, str(one_col[index + 0]))
                    num = num[:2]
                    number = float(num)
                    one_col[index] = number
            except ValueError:
                # print "not a number"
                not_float_index.append(index + 0)
        # trans not_float to np.nan
        for index in not_float_index:
            one_col[index] = np.nan

    train_selection.to_csv('../data/NumericData_clean_train.csv')



def make_wash_test():
    # wash test_set.csv
    test = pd.read_csv('../data/test_set.csv')
    log = open('../data/NumericData_tableID', 'r')
    lines = log.readlines()
    table_id_list = []
    for line in lines:
        table_id_list.append(str(line.replace('\n', '').replace('\r', '')))
    test_selection = test.ix[:, table_id_list]
    # change str to float
    test_selection = test_selection.convert_objects(convert_numeric=True)

    # solve the not-float data
    for table in table_id_list:
        print 'checking table : ' + str(table) + '...'
        not_float_index = []
        one_col = test_selection.ix[:, table]
        col_len = len(one_col)
        for index in range(col_len):
            # print index
            try:
                try:
                    number = float(one_col[index + 0])
                    if one_col.dtype != 'float64' and one_col.dtype != 'float32' and one_col.dtype != 'float':
                        one_col[index] = number
                except Exception, e:
                    num = filter(str.isdigit, str(one_col[index + 0]))
                    num = num[:2]
                    number = float(num)
                    one_col[index] = number
            except ValueError:
                # print "not a number"
                not_float_index.append(index + 0)
        # trans not_float to np.nan
        for index in not_float_index:
            one_col[index] = np.nan

    test_selection.to_csv('../data/NumericData_clean_test.csv')

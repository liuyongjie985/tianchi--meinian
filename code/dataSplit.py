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



def make_split():
    isServer = 0
    train = pd.read_csv('../data/train_set.csv')

    log = open('../data/NumericData_tableID', 'w')
    log1 = open('../data/categoricalData_tableID', 'w')
    log2 = open('../data/StringData_tableID', 'w')
    table_list = train.columns.values.tolist()
    flag = 0

    float_table_list = []
    categorical_table_list = []
    string_table_list = []

    for table in table_list:
        print 'checking table : ' + str(table) + '...'
        content_set = set()
        if flag < 6:
            flag += 1
            continue
        else:
            not_float_in_col_num = 0
            nan_num = 0
            not_float_index = []
            one_col = train.ix[:, table]
            col_len = len(one_col)
            for index in range(col_len):
                item = one_col[index]
                # print item
                content_set.add(item)
                try:
                    number = float(item)
                    try:
                        if np.isnan(number):
                            nan_num += 1
                    except Exception, e:
                        # if can't judge nan, it is not nan.
                        print e
                        print str(table) + '   ---nan---error---   ' + str(index)
                        continue
                except ValueError:
                    # print "not a number"
                    not_float_in_col_num += 1
                    not_float_index.append(index)
            not_float_percent = float(not_float_in_col_num) / (float(col_len) - float(nan_num))
            if not_float_percent <= 0.3:
                # this col is numeric
                float_table_list.append(table)
                # trans not_float to np.nan
                # for index in not_float_index:
                #     one_col[index] = np.nan
            else:
                unique_item_percent = float(len(content_set) - 1) / (float(col_len) - float(nan_num))
                if unique_item_percent <= 0.1:
                    # this col is categorical
                    categorical_table_list.append(table)
                else:
                    # this col is string
                    string_table_list.append(table)


    for table_id in float_table_list:
        log.write(str(table_id) + '\n')
    for table_id in categorical_table_list:
        log1.write(str(table_id) + '\n')
    for table_id in string_table_list:
        log2.write(str(table_id) + '\n')
    log.close()
    log1.close()
    log2.close()

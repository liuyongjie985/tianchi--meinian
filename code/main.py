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

import dataSplit
import dataWash
import ExactCatergoryLable
import trainAndTest
import clear_negative
import dataCleaning

import numpy as np
import pandas as pd
import os

# Limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

import warnings
# ignore annoying warning (from sklearn and seaborn)
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn



if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')

    # isFast is a special param, you can find more description in README.md
    isFast = 1

    # print '---------------------data re-build start------------------------'
    # use 'dataCleaning.py' to re-build data to 'train_set.csv' and 'test_set.csv'
    # do it when 'dataCleaning.py' is imported

    if isFast == 0:
        print '---------------------data split start------------------------'
        # use 'dataSplit.py' to find 3 types of data's tableID, and save to 'NumericData_tableID'
        # and 'categoricalData_tableID' and 'StringData_tableID'
        dataSplit.make_split()

        print '---------------------data wash NumericData start------------------------'
        # use 'dataWash.py' to wash the data in file 'NumericData_tableID'
        # and save washed data to new csv 'NumericData_clean_train.csv' and 'NumericData_clean_test.csv'
        dataWash.make_wash_train()
        dataWash.make_wash_test()
    else:
        os.system('cp ./files/* ../data/')
        time.sleep(10)

    # print '---------------------data wash and exact categoricalData start------------------------'
    # # use 'ExactCatergoryLable.py' to wash the data in file 'categoricalData_tableID'
    # # and save washed data to new csv 'CategoryData_clean_train.csv' and 'CategoryData_clean_test.csv'
    # # then save washed data y_train to new csv 'y_train_clear.csv'
    ExactCatergoryLable.exactCateroryLable()
    ExactCatergoryLable.clearYTrain()
    #
    # print '---------------------train and predict start------------------------'
    # # load 5 csv '', '', '', '' and '' to train 5 models
    # # and then predict the 5 types of result, and save to 5 files 'pred_1', 'pred_2', 'pred_3', 'pred_4' and 'pred_5'
    # # then merge the 5 files to the result csv 'submit_20180203_040506.csv'
    for i in range(5):
        trainAndTest.do_model(pred_type=i)
    trainAndTest.mergeResult()
    result_input_path = '../data/submit_20180506_000000_temp.csv'
    result_output_path = '../submit/submit_20180506_'+str(random.randint(100,1000))+'.csv'
    clear_negative.clearNegative(result_input_path, result_output_path)
    print 'all-ok'

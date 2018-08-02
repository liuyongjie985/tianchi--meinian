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
import warnings

import dataSplit
import dataWash
import trainAndTest

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew  # for some statistics
from subprocess import check_output

# %matplotlib inline 可以在Ipython编译器里直接使用，功能是可以内嵌绘图，并且可以省略掉plt.show()这一步。
# Matlab-style plotting
import matplotlib.pyplot as plt
color = sns.color_palette()
sns.set_style('darkgrid')

# ignore annoying warning (from sklearn and seaborn)
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

# Limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))



if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')

    # print '---------------------data re-build start------------------------'
    # # use 'dataCleaning.py' to re-build data to 'train_set.csv' and 'test_set.csv'
    #
    # print '---------------------data split start------------------------'
    # # use 'dataSplit.py' to find 3 types of data's tableID, and save to 'NumericData_tableID'
    # # and 'categoricalData_tableID' and 'StringData_tableID'
    # dataSplit.make_split()
    #
    # print '---------------------data wash start------------------------'
    # # use 'dataWash.py' to wash the data in file 'NumericData_tableID'
    # # and save washed data to new csv 'NumericData_clean_train.csv' and 'NumericData_clean_test.csv'
    # dataWash.make_wash_train()
    # dataWash.make_wash_test()
    #
    # print '---------------------train and predict start------------------------'
    # # load 5 csv '', '', '', '' and '' to train 5 models
    # # and then predict the 5 types of result, and save to 5 files 'pred_1', 'pred_2', 'pred_3', 'pred_4' and 'pred_5'
    # # then merge the 5 files to the result csv 'submit_20180203_040506.csv'
    # for i in range(5):
    #     trainAndTest.do_model(pred_type=i)



    # dataSplit.make_split() ------------------ ok!
    # trainAndTest.mergeResult() ------------------ ok!
    # dataWash.make_wash_train() ------------------ ok!
    # dataWash.make_wash_test() ------------------ ok!
    # ExactCatergoryLable.exactCateroryLable() ------------------ ok!
    # ExactCatergoryLable.clearYTrain() ------------------ ok!
    # trainAndTest.do_model(pred_type=i) ------------------ ok!
    # trainAndTest.mergeResult() ------------------ ok!

    print 'all-ok'


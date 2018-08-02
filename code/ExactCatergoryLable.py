# -*- coding: utf-8 -*-
import re
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import sys

import warnings
# ignore annoying warning (from sklearn and seaborn)
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

NAN = 'NAN'

# 格式 key&tableid

## 处理完的 tran_set.csv
## 处理完的 test_set.csv
## 这里要做路径修改
train_data = pd.DataFrame()
test_data = pd.DataFrame()


# 这个是 sklearn 中labelencoding 的dict 
ledict = {}
onlylabel_ledict = {}
catergory_tableidlist = []
def exact_keyword(table_id,keyword,train_data,ledict):
    key_word_1=keyword
    key = table_id
    feature_list = []
    text_key = list(train_data[key])
    for i in range(len(text_key)):
        tmp = re.split(r',|，|\.|。|；|:|：',str(text_key[i]))
        #print(tmp)
        result = NAN
        for item in tmp:
            if item.find(key_word_1)!=-1:
                result = item
                break
        feature_list.append(result)
    # encoding 特征
    le = preprocessing.LabelEncoder()
    le.fit(feature_list)
    transfor = le.transform(feature_list)
    name = '%s&%s' % (key_word_1,key)
    ledict[name] = le
    return transfor,name

def exactKeyWordFeather(keyword):
    
    tablecolms = train_data.columns.tolist()
    keyword_tableid = {}
    onetext = train_data.iloc[:5000]
    onetext = onetext.values.tolist()
    for item in keyword:
        keyword_tableid[item] = []
    # 获得 keyword --> tableidlist 列表
    # '血压': [001,002,003...]
    for one_item_text in onetext:
        for i in range(len(one_item_text)):
            for item in keyword:
                text1 = str(one_item_text[i])
                if text1.find(item)!=-1:
                    if tablecolms[i] not in keyword_tableid[item]:
                        keyword_tableid[item].append(tablecolms[i])
                    break
    # newfeature 得到新的feature
    newfeature = []
    featurename_list = []
    for key,table_id_list in keyword_tableid.items():
        for table_id in table_id_list:
            transfor,name = exact_keyword(table_id,key,train_data,ledict)
            newfeature.append(transfor)
            featurename_list.append(name)
    newfeature = np.array(newfeature)
    # 做一个翻转
    newfeature = newfeature.transpose()
    df =pd.DataFrame(newfeature,columns=featurename_list,index=list(train_data['vid']))
    df.index.name = 'vid'
    return df,keyword_tableid



### test 中 类别属性转换

def trans_keyword_test(table_id,keyword,test_data,ledict):
    key_word_1=keyword
    key = table_id
    feature_list = []
    text_key = list(test_data[key])
    name = '%s&%s' % (key_word_1,key)
    le = ledict[name]
    for i in range(len(text_key)):
        tmp = re.split(r',|，|\.|。|；|:|：',str(text_key[i]))
        #print(tmp)
        result = NAN
        for item in tmp:
            if item.find(key_word_1)!=-1:
                # 清理train 没出现过的item 
                # 这一步跟之前的不同
                if item in le.classes_:
                    result = item
                    break
        feature_list.append(result)
    # 从dict取 le
    transfor = le.transform(feature_list)
    ledict[name] = le
    return transfor,name

def getTestKeyWordFeather(keyword_tableid):
    newfeature = []
    featurename_list = []
    for key,table_id_list in keyword_tableid.items():
        for table_id in table_id_list:
            transfor,name = trans_keyword_test(table_id,key,test_data,ledict)
            newfeature.append(transfor)
            featurename_list.append(name)
    newfeature = np.array(newfeature)
    # 做一个翻转
    newfeature = newfeature.transpose()
    test_df =pd.DataFrame(newfeature,columns=featurename_list,index=list(test_data['vid']),)
    test_df.index.name = 'vid'
    return test_df

## 单纯根据提供的category 列名 label encoding
# 
def getCategoryLabel_fromtrain():
    # 格式 key&tableid
    newtrain_feature = []
    for tableid in catergory_tableidlist:
        le = preprocessing.LabelEncoder()
        tabletext = list(train_data[tableid])
        tabletext.append(NAN)
        le.fit(tabletext)
        onlylabel_ledict[tableid] = le
        # 把刚才加入的NAN pop 掉
        tabletext.pop()
        newtrain_feature.append(le.transform(tabletext))
    # 输出train
    newtrain_feature = np.array(newtrain_feature)
    newtrain_feature = newtrain_feature.transpose()
    train_df =pd.DataFrame(newtrain_feature,columns=catergory_tableidlist,index=list(train_data['vid']))
    train_df.index.name = 'vid'
    return train_df
def getCategoryLabel_fromtest():
    newtext_feature = []
    for tableid in catergory_tableidlist:
        le = onlylabel_ledict[tableid]
        newtext = []
        le_dict = {}
        for item in le.classes_:
            le_dict[item] = 1
        for item in test_data[tableid]:
            if item in le_dict:
                newtext.append(item)
            else:
                newtext.append(NAN)
        newtext_feature.append(le.transform(newtext))
    # 输出test
    newtext_feature = np.array(newtext_feature)
    newtext_feature = newtext_feature.transpose()
    test_df =pd.DataFrame(newtext_feature,columns=catergory_tableidlist,index=list(test_data['vid']))
    test_df.index.name = 'vid'
    return test_df


### 
### 作为主函数
def exactCateroryLable():
    global train_data
    global test_data
    global catergory_tableidlist
    train_data = pd.read_csv("../data/train_set.csv")
    test_data = pd.read_csv("../data/test_set.csv")
    fin = open('../data/categoricalData_tableID', 'r')
    catergory_tableidlist = fin.readlines()
    catergory_tableidlist = [line.strip() for line in catergory_tableidlist if line.strip() != ""]

    ##key word
    keyword = ['血压','血脂','血糖','心肌梗塞','血管弹性','心率','糖尿','脂肪','尿酸','窦性']
    df_train_keyword,keyword_tableid = exactKeyWordFeather(keyword)
    df_test_keyword = getTestKeyWordFeather(keyword_tableid)
    df_train_catergory = getCategoryLabel_fromtrain()
    df_test_catergory = getCategoryLabel_fromtest()
    all_train = pd.merge(df_train_keyword,df_train_catergory,left_index=True,right_index=True)
    all_test = pd.merge(df_test_keyword,df_test_catergory,left_index=True,right_index=True)

    ## 输出
    all_train.to_csv('../data/CategoryData_clean_train.csv')
    all_test.to_csv('../data/CategoryData_clean_test.csv')


def clearYTrain():
    ## 对train中的 回归值y 去燥
    # 主要是去除异常点
    colmnames = ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']
    values = {}
    for item in colmnames:
        values[item] = train_data[item].mean()
    # 对 y 值中的空值用均值填充
    newtextdata = train_data.fillna(value=values)
    y_train = np.array(newtextdata[colmnames])
    for i in range(y_train.shape[0]):
        for j in range(y_train.shape[1]):
            if y_train[i][j]<=0:
                y_train[i][j] = np.mean(y_train[:,j])
            if y_train[i][j]>7*np.mean(y_train[:,j]):
                #print "第%d列 %f" %(j,y_train[i][j])
                y_train[i][j] = 2*np.mean(y_train[:,j])
                
    df = pd.DataFrame(y_train,columns=colmnames)
    df.to_csv('../data/y_train_clear.csv',index=False)

# -*- coding: utf-8 -*-
import numpy as np
import sys
# 去负数脚本
# python 3
# python clearnegativeitem.py filename
# 避免出现负数，词脚本
# 同时处理数据为三位小数
def clearNegative(filein,fileout):
    fin = open(filein,'r')
    result = []
    vid = []
    for line in fin:
        line = line.strip()
        if line =="":
            continue
        tmp = line.split(',')
        if len(tmp) < 6:
            print("数据出错!!!")
            exit(0)
        result.append(tmp[1:])
        vid.append(tmp[0])
    # 计算均值
    result = np.array(result)
    result = result.astype('float')

    mean_every = []
    for i in range(result.shape[1]):
        newlist = [item for item in result[:,i] if item>0]
        mean_every.append(sum(newlist)/len(newlist))

    # 均值填充负数
    for i in range(len(result)):
        for j in range(len(result[i])):
            if result[i][j]<0:
                print(result[i][j])
                result[i][j]=mean_every[j]
    fout = open(fileout,'w')
    result = list(result)
    for i in range(len(vid)):
        tmp = []
        tmp.append(vid[i])
        tmp.extend([round(item,3) for item in result[i]])
        fout.write(",".join([str(v) for v in tmp]))
        fout.write('\n')
    fout.close()
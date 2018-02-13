# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:25:35 2018

@author: Everyheart
"""


#这里的目标是找到第一次出现的元素，并且将与其一样的后面几个元素求和得到结果
#这里只需要将数据保存在X中，然后运行这个代码，结果会显示在result里。
X=[3,1,2,2,3,3,4,5,6,6,6,6,7,5,4,2,2,1]
def dealdata(X):
    Result=[]
    first=[]
    for i in range(len(X)):
        if X[i] in first:
            continue
        else :
            first.append(X[i])
            qiuhe=0
            for s in range(i,len(X)):
                if X[s]==X[i]:
                    qiuhe+=X[s]
                else:
                    break
            Result.append(qiuhe)
    return Result

result=dealdata(X)
#最后的result[3,1,4,4,5,24,7]
    
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 21:09:55 2018

@author: Everyheart
"""
from data import data1,data2,plotcluster,plotclustertog,leibiecishu,generatepair,compute,findprob,generatedata,gendatedata,Logisticdata
import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing

#这里的主要目的是为了直接实现对于代码的训练
A=[600000,600010,600015,600016,600019,600028,600030,600031,600036,600048,600050,600104,600111,600123,600256,600348,600362,600383,600489,600518,600519,600547,600549,600585,600837,600887,600999,601006,601088,601166,601169,601288,601299,601318,601328,601336,601398,601601,601628,601668,601669,601688,601699,601766,601788,601818,601857,601899,601901,601989]  

#
#%%
df_day,num_price,df=data1(code='000001',asset='INDEX',start_date='2015-10-01',end_date='',freq='30min')
#df_day.to_csv('c:/Users/Everyheart/database/000001.csv')#保存数据用
#price=data2(df_day,'price',num_price)

#%
#这个是分别画出各个分类的走势的图，如果希望可以单独看哪一个图
#k_center=kmeans.cluster_centers_
#for i in range(n_clusters):
#    c=plotcluster(k_center,num_price,n=i)
#下面我们需要画出整合在一起的图
#%%
#下面需要进行的是对这里的数据进行拓展，也就是对于data2中的数据拓展加成交量
#下面是加入了成交量后得到的分类的图

lx='close'#这里可以设置成'price&vol''close'，'price'来分别对于不同的
n_clusters=49#分类的数量，可以调
random_state1=200#这里是K-means中初始的随机设置


pricevol=data2(df_day,lx,num_price)#这里对于初始的数据进行处理实现
#pricevol_sacle=preprocessing.scale(pricevol)
price_sacle=pricevol
kmeans=KMeans(n_clusters=n_clusters,random_state=random_state1).fit(price_sacle)#这里利用sklearn的kmeans来实现分类
c=kmeans.labels_#这里包含具体的类
js=leibiecishu(c,n_clusters)#这里是获得所有分类的计数
k_center_vol=kmeans.cluster_centers_#这里获得所有类型的均值的情况
plotclustertog(k_center_vol,num_price,n_clusters,js,1,lx)#这里是将各个类的净值图，个数画出来
#备注，这里可以用最新几个交易日的走势看c最后几个的分类，在图中找到类的图片，进行对比

#%%
#根据是的数据生成一个可以用来计算转移阵的数据生成数据的函数
T=generatepair(c,num_price=num_price,n_clusters=8)#这里是对日与日之间的关系进行处理的，得到转移的对
#-------------------------------------------------
data1,T0,T1,k_center,k_means=generatepair(pricevol,num_price=num_price,n_clusters=8,train=1)#这里输入原始的数据，然后对于日内的情况进行分类
#日内的分类后，对于日内的数据，进行处理，得到了T0是分类后，原来每一天的类，T1是生成的对，k_center是这里对于日，
#每一个周期的数据进行分类的类的中心，而n_clusters=8代表分成多少个中心，data1就是原来周期的数据
#---------------------------------------------
#生成数据后的是日线的是T，分钟的是T0
#接下来，需要对于这里的矩阵进行处理，实现对于转移概率的计算
Q1,JS1,P1,S1=compute(T1)#这里的Q1是转移数量，JS是每一个状态作为起始状态的次数，P1是转移概率阵，S1是转移的数量+概率阵
Q,JS,P,S=compute(T)

#接下来，我们需要知道当前状态，查看之后状态可能状态和概率，

D=findprob(S1,state_i=3)#这是给定state_i为起始阵，查看转移到各个状态的概率

#%%
#接下来我们要做的，就是将A50的数据整理在一起，并且对比不同时间和不同的股票的转移阵，以及对应的正确性

#（1）首先我们先做出一个函数可以计算出这里的数据获得，1.不同时间的，2.不同的股票的数据

A=[600000,600010,600015,600016,600019,600028,600030,600031,600036,600048,600050,600104,600111,600123,600256,600348,600362,600383,600489,600518,600519,600547,600549,600585,600837,600887,600999,601006,601088,601166,601169,601288,601299,601318,601328,601336,601398,601601,601628,601668,601669,601688,601699,601766,601788,601818,601857,601899,601901,601989]  
#这里的A是A50的指数
X1,X2=generatedata(A[:10],lx='price',start_date='2017-12-01',end_date='',freq='30min',asset='E',ty=2,ratio=0.3)

#这里的X1,X2为两组数据，第一组是0.7的比例的股票，或者0.7比例的时间为界划分得到的。X2为第二部分
#asset='E' ty=1代表按照股票划分ratio为界  ty=2代表按照时间划分ratio为界

#%%
#接下来我们就要分别的去求出对应的转移阵
hb,X1,X2=generatedata(A[:50],lx='price',start_date='2015-12-01',end_date='',freq='30min',asset='E',ty=2,ratio=0.3)
#%%
#用X1生成这里的分类方法，得到转移阵
num_price=8#这个看具体的freq的设置来设置
data1,T0,T1,k_center1,k_means=generatepair(X1,num_price=num_price,n_clusters=8,train=1)
Q1,J1,P1,S1=compute(T1)
#利用上面的kmeans对于下面的情况生成最后的结果
data2,T0_1,T1_1,k_center1,k_means=generatepair(X2,num_price=num_price,n_clusters=8,train=2,kmeans=k_means)
Q2,J2,P2,S2=compute(T1_1)

#%%
#上面已经完成了时间维度和股票维度的比较，目前看，对于这里的就是对于日线的转移阵的处理

#（1)这里的数据依然要基于X1,X2
#首先我们用X1的数据，做日线的分类
A=[600000,600010,600015,600016,600019,600028,600030,600031,600036,600048,600050,600104,600111,600123,600256,600348,600362,600383,600489,600518,600519,600547,600549,600585,600837,600887,600999,601006,601088,601166,601169,601288,601299,601318,601328,601336,601398,601601,601628,601668,601669,601688,601699,601766,601788,601818,601857,601899,601901,601989]  

hb,X1,X2=generatedata(A[:50],lx='price',start_date='2015-12-01',end_date='',freq='30min',asset='E',ty=2,ratio=0.3)
#%%
label,q,s,p,p2,p22,k_center_data=gendatedata(X1,X2,hb,random_state1=200,n_clusters=8,num_price=8)

#%%
X=np.append(X1,X2,axis=0)


X,Y,socre=Logisticdata(8,T0)#这里实现了预测，用交叉验证集来验证，最后得到了得分。这里结果基本都在30%左右，分类这里是8类的情况




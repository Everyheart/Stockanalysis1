# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:35:07 2018

@author: Everyheart
"""
#（1）引入库和函数，所有的函数都在data.py中
from data import data1,data2,plotcluster,plotclustertog,leibiecishu,generatepair,compute,findprob,generatedata,gendatedata,Logisticdata
import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
#设置A为上证50成分的
A=[600000,600010,600015,600016,600019,600028,600030,600031,600036,600048,600050,600104,600111,600123,600256,600348,600362,600383,600489,600518,600519,600547,600549,600585,600837,600887,600999,601006,601088,601166,601169,601288,601299,601318,601328,601336,601398,601601,601628,601668,601669,601688,601699,601766,601788,601818,601857,601899,601901,601989]  


#%%

#第一部分：基于30分钟级别的数据对于交易日进行分类
#------------------------------------------------------------------------
#（2）调用data1，获得数据，变成DATAFrame
df_day,num_price,df=data1(code='000001',asset='INDEX',start_date='2015-10-01',end_date='2017-01-01',freq='30min')
#Tips：（1）函数的目标是将这段时间内，如果提取的是30分钟数据，在多行，我们将所有的同一天的样本集中到一行中，并且对于收盘价改为收盘收益率
#      （2）df_day,是一个n(day)*k的矩阵，每一行分别是该交易日的所有K线数据，而列为该交易日的各种指标，前5列是这个交易日的指标，后面分别是每一个freq的指标。
#      （3）num_price：是由输入freq决定的，一个交易日有4个小时，30分钟就是8,60分钟就是4，以此类推‘
#      （4）df：df代表的是原始的数据，也就是每一个freq为一行，列为5中指标，分别是：开盘，收盘，最高，最低，成交量，成交额
#综上所述，这里的data1函数就是实现数据的提取，并且将数据变成表格，便于后面的使用


#（3）设置相关的参数：
lx='close'#这里可以设置成'price&vol''close'，'price'来分别对于不同的；类型的数据进行处理
n_clusters=49#分类的数量，这里是指将不同交易日进行分类，分成49类，分类是基于该交易日的freq级别的lx类型的数据来分类的
random_state1=200#这里是K-means中初始的随机设置

#（4）用data2 将数据变成矩阵形式
pricevol=data2(df_day,lx,num_price)#这里对于初始的数据进行处理得到一个矩阵，包含我们想要的数据，矩阵为m(samples)*n(features)
#pricevol_sacle=preprocessing.scale(pricevol)#这一步是数据的处理，可以省略
price_sacle=pricevol
#（5）利用Kmeans将数据分类
kmeans=KMeans(n_clusters=n_clusters,random_state=random_state1).fit(price_sacle)#这里利用sklearn的kmeans来实现分类
c=kmeans.labels_#这里包含具体的类，这里的类基于训练的数据，得到的各个样本的类
js=leibiecishu(c,n_clusters)#这里是获得所有分类的计数，得到一个list，分别有各个类别的数量
k_center_vol=kmeans.cluster_centers_#这里获得所有类型的均值的情况
#（6）画出分类后均值的图形
plotclustertog(k_center_vol,num_price,n_clusters,js,1,lx)#这里是将各个类的净值图，个数画出来
#备注，这里可以用最新几个交易日的走势看c最后几个的分类，在图中找到类的图片，进行对比


#%%
#-------------------------------------------------------
#第二部分：基于数据求得概率转移阵
#------------------------------------------------------

#(1)根据是的数据j生成一个可以用来计算转移阵的数据生成数据的函数
T=generatepair(c,num_price=num_price,n_clusters=8)#这里是对日与日之间的关系进行处理的，得到转移的对
#-------------------------------------------------
data1,T0,T1,k_center,k_means=generatepair(pricevol,num_price=num_price,n_clusters=8,train=1)#这里输入原始的数据，然后对于日内的情况进行分类
#日内的分类后，对于日内的数据，进行处理，得到了T0是分类后，原来每一天的类，T1是生成的对，k_center是这里对于日，
#每一个周期的数据进行分类的类的中心，而n_clusters=8代表分成多少个中心，data1就是原来周期的数据
#---------------------------------------------
#函数generatepair：这个函数的设置是为了实现将原本的一个日线的状态（类）的列，变成[a,b]这种从状态a到状态b的这样的一对数。或者将一个初始的price类型
#-也就是经过data2处理得到的数据price，将其进行分类，对于各个30分钟的的状态的转移生成[a,b]这样的Pair
#该函数具体的内容请输入：help(generatepair)






#生成数据后的是日线的状态对是T，分钟的是T0

#（2）接下来，需要对于这里的矩阵进行处理，实现对于转移概率的计算，转移阵和
Q1,JS1,P1,S1=compute(T1)#这里的Q1是转移数量，JS是每一个状态作为起始状态的次数，P1是转移概率阵，S1是转移的数量+概率阵
Q,JS,P,S=compute(T)
#(3)接下来，我们需要知道当前状态，查看之后状态可能状态和概率，
D=findprob(S1,state_i=3)#这是给定state_i为起始阵，查看转移到各个状态的概率


#%%
#-------------------------------------------------------
#第三部分：
#part1:接下来我们要以上证50为例，去分别统计，不同股票的转移概率是否一致
#------------------------------------------------------

#首先通过generatedata将数据分成两个部分。

hb,X1,X2=generatedata(A[:50],lx='price',start_date='2015-12-01',end_date='',freq='30min',asset='E',ty=1,ratio=0.3)
    #输入lx,起始结束时间，股票代码，以及资产类型和数据划分比例，0.3代表7:3，ty表示数据分割类型，ty=1代表依据股票代码分割，ty=2代表依据交易时间分割
    #这里的输出X1，X2分别代表数据分割的前半部分和后半部分，而hb代表着所有获得的数据的列表。
#%% 这个分割号为了两个分开算，因为上一个获得数据的计算时间非常长
#用X1生成这里的分类方法，得到转移阵
num_price=8#这个看具体的freq的设置来设置
data1,T0,T1,k_center1,k_means=generatepair(X1,num_price=num_price,n_clusters=8,train=1)
Q1,J1,P1,S1=compute(T1)#这里的第一部分设置train=1,也就是进行分类，然后得到分类的Kmeans函数,
                       #在对第二部分数据进行处理的时候设置train=2,保持二者分类的一致性
#利用上面的kmeans对于下面的情况生成最后的结果
data2,T0_1,T1_1,k_center1,k_means=generatepair(X2,num_price=num_price,n_clusters=8,train=2,kmeans=k_means)
Q2,J2,P2,S2=compute(T1_1)


#%%
#-------------------------------------------------------
#第三部分：
#part2:接下来我们要以上证50为例，去分别统计，不同时间的的转移概率是否一致
#------------------------------------------------------
#首先通过generatedata将数据分成两个部分。

hb,X1,X2=generatedata(A[:50],lx='price',start_date='2015-12-01',end_date='',freq='30min',asset='E',ty=2,ratio=0.3)
    #输入lx,起始结束时间，股票代码，以及资产类型和数据划分比例，0.3代表7:3，ty表示数据分割类型，ty=1代表依据股票代码分割，ty=2代表依据交易时间分割
    #这里的输出X1，X2分别代表数据分割的前半部分和后半部分，而hb代表着所有获得的数据的列表。
#%% 这个分割号为了两个分开算，因为上一个获得数据的计算时间非常长
#用X1生成这里的分类方法，得到转移阵
num_price=8#这个看具体的freq的设置来设置
data1,T0,T1,k_center1,k_means=generatepair(X1,num_price=num_price,n_clusters=8,train=1)
Q1,J1,P1,S1=compute(T1)#这里的第一部分设置train=1,也就是进行分类，然后得到分类的Kmeans函数,
                       #在对第二部分数据进行处理的时候设置train=2,保持二者分类的一致性
#利用上面的kmeans对于下面的情况生成最后的结果
data2,T0_1,T1_1,k_center1,k_means=generatepair(X2,num_price=num_price,n_clusters=8,train=2,kmeans=k_means)
Q2,J2,P2,S2=compute(T1_1)


#%%

#-------------------------------------------------------
#第四部分：对于日线的数据转移的马氏性进行研究，也可以分成两个部分，这里我们只展示第一部分，也就是不同的股票的日线转移是否具有马氏性
#而对于不同的时间的，只用调整起始时间，ratio改成0，就可以获得不同的数据，计算出转移阵了
#------------------------------------------------------

#基于上面的函数获得数据
hb,X1,X2=generatedata(A[:50],lx='price',start_date='2015-12-01',end_date='',freq='30min',asset='E',ty=2,ratio=0.3)
#这里可以得到两个部分的股票的数据
label,q,s,p,p2,p22,k_center_data=gendatedata(X1,X2,hb,random_state1=200,n_clusters=8,num_price=8)


#%%
#-------------------------------------------------------
#第一部分：基于数据前面的数据去预测出后面的数据，最后8种分类的准确率是30%左右
#------------------------------------------------------

X,Y,socre=Logisticdata(8,T0)#这里实现了预测，用交叉验证集来验证，最后得到了得分。这里结果基本都在30%左右，分类这里是8类的情况








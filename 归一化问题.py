# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 20:40:27 2018

@author: Everyheart
"""

import numpy as np

#===================================
#思路一 基于sigmoid的函数的衍生
#==================================
#函数一：在sigmoid 函数进行调整k来缩放

def sigmoid_1(x,k):
    '''
    输入x:一个n*1的矩阵
    输入k为缩放比例，可以通过调整缩放比例来实现对于数据收缩的调整
    '''
    u=np.mean(x)#获得均值
    Y=[]
    for i in range(len(x)):
        c=x[i]
        y=1/(1+np.exp(-k*(c-u)))
        Y.append(y)
    return Y

#函数二： 
def sigmoid_2(x,t=0.99):
    '''
    输入x：一个n*1的矩阵
    
    令点P为(max-min)/2+u，这里表示最大最小值距离均值的平均距离加上均值的点
    输入t：因为sigomid函数容易导致数据趋近于0或者1，t表示的点P的转化后的位置，
    t我们可以设置为0.99,0.999,或0.9999来防止其过度趋向于1，如果单纯调整K可能会导致与其接近的数据转化后的数据倾向于1，数据之间的差距太小。
    '''
    Y=[]
    max_1=max(x)
    min_1=min(x)
    u=np.mean(x)
    for i in range(len(x)):
        c=x[i]
        k=np.log((1/t)-1 )/((min_1-max_1)/2)
        y=1/(1+np.exp(-k*(c-u)))
        Y.append(y)
    return Y

#===================================
#思路二 基于sin的函数的衍生
#==================================
#函数三：
def sin_1(x):
    Y=[]
    for i in range(len(x)):    
        
        X_std=(x[i]-min(x))/(max(x)-min(x))#将数据缩放到[-pie/2,pie/2]间
        X0=np.pi*X_std-np.pi/2
        
        X1=0.5*np.sin(X0)+0.5#数据带入sin函数中
        Y.append(X1)
    return Y

#===========================================================
#构建一组含有100个介于[-100,100]的随机数
x=[]
x1=[]
for i in range(1,100):
    c=np.random.randint(-100,100)    
    x.append(c)
    x1.append([c])
    
#原本做线性处理到[0,1]数据的方差
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(x1))
Y0=scaler.transform(x1)
print('线性的方差',np.var(Y0))

#=========================
#函数一
Y=sigmoid_1(x,k=2)
print('函数一处理后的方差',np.var(Y))
#函数二
Y1=sigmoid_2(x,t=0.99)
print('函数二处理后的方差',np.var(Y1))
#函数三：
Y2=sin_1(x)
print('函数三处理后的方差',np.var(Y2))
#======
#最后输出的结果如下：

#线性的方差 0.09082363922892163
#函数一处理后的方差 0.24822024525272038
#函数二处理后的方差 0.1589382490128692
#函数三处理后的方差 0.136265754099373

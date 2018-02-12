# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 21:11:20 2018

@author: Everyheart
"""

#这里是需要实现的是对于数据的构建
#%%
import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt



def data1(code='000001',asset='INDEX',start_date='2017-10-01',end_date='2018-01-01',freq='30min'):
    '''
    输入所需要的股票代码，类型，起始和结束日期，和周期，将同一天的数据，放入列表的同一行中
    code 是股票代码，结合asset使用
    当为股票时asset为E,指数为INDEX，
    freq可以是'60min''30min''15min''5min'
    输出：
    #Tips：（1）函数的目标是将这段时间内，如果提取的是30分钟数据，在多行，我们将所有的同一天的样本集中到一行中，并且对于收盘价改为收盘收益率
    #（2）df_day,是一个n(day)*k的矩阵，每一行分别是该交易日的所有K线数据，而列为该交易日的各种指标，前5列是这个交易日的指标，后面分别是每一个freq的指标。
    #（3）m：是由输入freq决定的，一个交易日有4个小时，30分钟就是8,60分钟就是4，以此类推‘
    #（4）df：df代表的是原始的数据，也就是每一个freq为一行，列为5中指标，分别是：开盘，收盘，最高，最低，成交量，成交额
    #综上所述，这里的data1函数就是实现数据的提取，并且将数据变成表格，便于后面的使用
    '''
    cons = ts.get_apis()
    df = ts.bar(code, conn=cons, freq=freq,asset=asset,start_date=start_date, end_date=end_date)#获取数据表格
    df=df.sort_index(axis=0,ascending=True)
    length1=len(df)#获取表格的长度，在未来多组数据合并的时候，方便加上指数变化特征。
    #接下来我们需要将数据汇总到一个表格中，这个表格刚好就是日级别的
    df_day = ts.bar(code, conn=cons, freq='D',asset=asset,start_date=start_date, end_date=end_date)
    df_day=df_day.sort_index(axis=0,ascending=True)
    
    if freq=='60min':
        m=4
    elif freq=='30min':
        m=8
    elif freq=='15min':
        m=16
    elif freq=='5min':
        m=48
    #df_day=df_day.ix[:,0]
    df_day=df_day.iloc[-int(length1/m):,:]
    length2=len(df_day)
    for t in range(m):#这里是构建出相应的列
        df_day['open'+str(t)]=np.zeros([length2,1])
        df_day['close'+str(t)]=np.zeros([length2,1])
        df_day['high'+str(t)]=np.zeros([length2,1])
        df_day['low'+str(t)]=np.zeros([length2,1])
        df_day['vol'+str(t)]=np.zeros([length2,1])
        df_day['amount'+str(t)]=np.zeros([length2,1])


    for i in range(0,length1,m):#将构建的列填充上相应的数据，
        a=int(i/m)
        for t in range(m):
            s=t#这里的ix方法已经不推荐了，因此我们用更稳定的方法去做
            df_day.loc[df_day.index[a],'open'+str(t)]=df.loc[df.index[i+s],'open']
            openprice=df.loc[df.index[i+s],'open']
            df_day.loc[df_day.index[a],'close'+str(t)]=(df.loc[df.index[i+s],'close']-openprice)*100/openprice
            df_day.loc[df_day.index[a],'high'+str(t)]=(df.loc[df.index[i+s],'high']-openprice)*100/openprice
            df_day.loc[df_day.index[a],'low'+str(t)]=(df.loc[df.index[i+s],'low']-openprice)*100/openprice
            df_day.loc[df_day.index[a],'vol'+str(t)]=df.loc[df.index[i+s],'vol']/(10*df_day.loc[df_day.index[a],'vol'])#这里的数字没错，但是在数量级上总的量数量级不一样
            df_day.loc[df_day.index[a],'amount'+str(t)]=df.loc[df.index[i+s],'amount']/df_day.loc[df_day.index[a],'amount']
    return df_day,m,df
    


#df_day,num_price=data1(code='000001',asset='INDEX',start_date='2017-10-01',end_date='2018-01-01',freq='30min')#这里经常执行出问题，可以用try_except来优化

#%
def data2(df,leixin,num_price):
    #这个函数的目的是将上面的数据转化成我们想要的特征，这里需要构建一个可以选择的特征，最后形成可以直接输出的X来进行相应的聚类。
    '''
    将原来的列表转化成一个矩阵，并且根据不同的leixin，存放不同的数据进去
    df:输入的完整的表格
    leixin：从这个表格中搜集哪些数据
    num_price:代表这个时间周期如30分钟，一天中有几个，如8个
    输出值：price：这个是按照时间先后，排列的（close,high,low）的收益率
    '''
    if leixin=='price':#如果仅仅只有价格也就是各个时间段的三个涨幅
        pricename=[]
        for i in range(num_price):
            pricename.append('close'+str(i))
            pricename.append('high'+str(i))
            pricename.append('low'+str(i))
        df_price=df[pricename]
        price=df_price.values#这里返还出一个只有价格的矩阵
    elif leixin=='price&vol':
        pricename=[]
        for i in range(num_price):
            pricename.append('close'+str(i))
            pricename.append('high'+str(i))
            pricename.append('low'+str(i))
            pricename.append('vol'+str(i))
            pricename.append('amount'+str(i))
        df_price=df[pricename]
        price=df_price.values#这里返还出一个只有价格的矩阵
    elif leixin=='vol':
        pricename=[]
        for i in range(num_price):
            pricename.append('close'+str(i))
            pricename.append('vol'+str(i))
            pricename.append('amount'+str(i))
        df_price=df[pricename]
        price=df_price.values#这里返还出一个只有价格的矩阵     
    elif leixin=='close':
        pricename=[]
        for i in range(num_price):
            pricename.append('close'+str(i))
        df_price=df[pricename]
        price=df_price.values#这里返还出一个只有价格的矩阵           
    return price
#price=data2(df_day,'price',num_price)
            
#%
#下面定义一个将输出的cluster画图的


def plotcluster(price,num_price,n=1):
    '''
    这里只画出收盘价
    '''
    closelist=[]
    for i in range(num_price):
        closelist.append(3*i)
    close=price[:,closelist]
    plot_close=close[n]
    plt.figure()
    plt.title('cluster_center'+str(n))
    plt.plot(plot_close,'.-')

    plt.grid()
    
    return close
def plotclustertog(price,num_price,n,js,htlx=1,lx='price'):
    '''
    这是输入：
    price：由kmeans得到的特征的均值情况，有n_cluster个，
    num_price是指30分钟就是8，
    n是分类的个数，
    js是每一个类当中，各类的个数的计数list
    htlx=1时表示的是按照净值画的图，
    htlx!=1是收益率的图。
    lx,就是和上面的类型是一致的
    图中的列表分别代表：第几个类型；发生了几次，发生的比率
    '''
    closelist=[]
    
    for i in range(num_price):
        if lx=='close':
            closelist.append(1*i)
        elif lx=='price':
            closelist.append(3*i)
        elif lx=='price&vol':
            closelist.append(5*i)
    close=price[:,closelist]   
    
    pn=int(n**(1/2))#这里代表需要画图的数量的开方
    plt.figure()
    
    for i in range(n):
        #k=pn*100+pn*10+i+1
        if htlx==1:#这里就是按照净值画的图
            closepricebilv=close[i]/100
            closeprice=[]
            cp=1
            for s in range(close.shape[1]):
                cp=cp*(1+closepricebilv[s])
                closeprice.append(cp)
        
            plt.subplot(pn,pn,i+1).plot(closeprice,'.-',label=[i,js[i],int(js[i]*100/sum(js))])
            plt.grid()
            plt.legend()
        else:#这里就是原来的直接画图这里的收益率走势图，二者是对应的
            plt.subplot(pn,pn,i+1).plot(close[i],'.-',label=[i,js[i],int(js[i]*100/sum(js))])
            plt.grid()
            plt.legend()

#c=plotcluster(price,num_price,)


  
def leibiecishu(c,n_clusters):
    '''
    这个类别次数，是将所有类别的C，计数，统计得到，每一个类别的发生次数，
    c:类别的label
    n_clusters：是类别数量
    '''
    #m=len(c)
    js=[]
    for i in range(n_clusters):
        num=sum(c==i)
        js.append(num)
    return js
        

def generatepair(data,num_price=8,n_clusters=8,train=1,kmeans=0):
    '''
    #函数generatepair：这个函数的设置是为了实现将原本的一个日线的状态（类）的列，变成[a,b]这种从状态a到状态b的这样的一对数。或者将一个初始的price类型
    #-也就是经过data2处理得到的数据price，将其进行分类，对于各个30分钟的的状态的转移生成[a,b]这样的Pair
    
    这个函数可以面对两种类型的输入：
    类型1.日与日的数据，类型是1*n的数据，数据中是各种类型的state，也就是各种分类的状态
    类型2.输入一个price阵，可以是各种类型的价格成交量阵，这里的输入是：m*(num_price*l)这里用Kmeans分别对于各个进行分类，
    
    参数：
    数据data：这里的数据可以是n*1的日线的分类数据，也可以是n*m的一个分钟的数据集合。
    类型lx,这个输入后来可以省略，没有用到
    n_clusters kmeans的分类个数
    train=1,表示对于输入的数据进行训练分类，train=2,我们这里需要输入kmeans参数，这里的Kmeans可以是基于train=1时训练的输出的结果
    num_price：由data1函数产生的时间段，30分为8
    
    
    输出1：T：最后输出一个m*2的矩阵，分别是各个数列的
    输出2：T0：一个日的类别的矩阵，可以看到日线的矩阵，结构是m*num_price，比如20个样本，每日8个时段，就是20*8
           输出2是针对输入的是第二种类型的price阵：T0
    输出3：k_center:kmeans的类别的中心的情况,这个矩阵的第一列就是close的均值情况。
    输出4：data1：这个是将原来的矩阵恢复成以30分钟或者5分钟等为结构的矩阵,在别的函数中可能作为输入
    
    
    '''
    try:    
        k=data.shape[1]
    except:
        k=1
        
    if k==1:#这个是输入情况1
        m=data.shape[0]
        T=np.zeros([m-1,2])
        for i in range(m-1):
            T[i,0]=int(data[i])
            T[i,1]=int(data[i+1])
        return T
    
    else:
        #首先定义l,这个l是各个类型所有的变量数。
        #if lx=='price':
        #    l=3
        #elif lx=='price&vol':
        #    l=5
       # elif lx=='close':
        #    l=1
        #elif lx=='vol':
        #    l=3
        l=int(k/num_price)#这里的l的数量可以直接计算得到
        m0=data.shape[0]
        #接下来我们需要将这里的数据矩阵，变成一个分开的,这里用reshape就可以做到
        data1=data.reshape(int(m0*num_price),l)
        #接下来需要进行分类，得到类别
        from sklearn.cluster import KMeans
        
        if train==1:
            kmeans=KMeans(n_clusters=n_clusters,random_state=0).fit(data1)
            c=kmeans.labels_#这里的c虽然是按照顺序的，但是我们只研究日内的，因此我们只考虑num_price个独立开来的
        elif train==2:
            c=kmeans.predict(data1)
        k_center=kmeans.cluster_centers_
        #接下来将c由一列，变成我们想要的一个矩阵，矩阵长度为m0,矩阵宽度是num_price
        T0=c.reshape(m0,num_price)#行为样本数量，列为各个时间段的不同的状态
        
        #接下来，需要将T0分别列出来，变成一个m*2的阵，这里的提取原则是将T0中，各行构成num_price-1个
        m=m0*(num_price-1)
        T=np.zeros([1,2])
        t=np.zeros([num_price-1,2])

        for s in range(m0):
            
            for i in range(num_price-1):
                t[i,0]=T0[s,i]
                t[i,1]=T0[s,i+1]
            T=np.append(T,t,axis=0)#这里是将每一行提取出的num_price-1个的组合并入总的T阵中，最后输出T
        T=T[-m:,:]
        return data1,T0,T,k_center,kmeans
                
#data1,T0,T1,k_center=generatepair(pricevol,8)





#%
def compute(T,n_state=49,lx=1):
    '''
    这里的目标是输入m*2的转移的方式T,计算出T中各个状态转移的概率
    返还一个矩阵P，矩阵为n*n,第i行j列的元素就是第i个状态转移到第j个状态的概率，
    返还一个矩阵Q，有次数
    
    n_state:因为在后面的计算中会遇到特别的情况，我们加上给定状态数目，保持后面的一致性
    lx=1那么就和之前一样，默认一样。只有在gendatedata后面的函数中用到第二种情况，并且设定mt的个数为状态个数，因为在后面，有的状态可能没有数值，为了保证矩阵大小一致，可以做加法，而在这里设定mt，为分类数,和类型
    
    Q阵为转移的次数，行为起始点，列为结束状态，数值的转移的结果
    JS为计数的个数
    PQ阵里面既有次数，也有概率，这里的次数记录在整数位，概率记录在小数，如2.105就是说，发生了2次，概率是10.5%
    '''
    if lx==1:
        mt=int(T.max()+1)#这一步计算出了T中状态的个数
    elif lx==2:
        mt=n_state
    m=len(T)
    #接下来需要统计每一个状态转移到下一个状态的数据
    JS=np.zeros([mt,1])#这个是一个计数阵，记录每一个类型的发生的次数
    Q=np.zeros([mt,mt])
    for i in range(mt):
        q=0#计数用
        for s in range(m):
            if T[s,0]==i:
                q=q+1
                zyjg=int(T[s,1])#这个是转移后的状态
                Q[i,zyjg]=Q[i,zyjg]+1
        #计数的JS
        JS[i]=int(q)
    #接下来求每一个转移的概率，这里的概率就是每一个发生的次数比上计数的个数
    #P=Q
    P=np.zeros([mt,mt])
    for i in range(mt):
        if JS[i]!=0:
            P[i]=Q[i]/JS[i]
    PQ=np.zeros([mt,mt])
    for i in range(mt):
        if JS[i]!=0:
            for s in range(mt):
                PQ[i,s]=Q[i,s]+Q[i,s]/JS[i]

    
    return Q,JS,P,PQ
#Q,JS,P,S=compute(T)            
                
                
        


def findprob(T,state_i=1):
    '''
    这里的T是输入的矩阵，可以是数量，也可以是概率阵,也可以是汇总的，
    这里将矩阵第state_i行中非0元素汇总并且加上这些数据所在的列的序列
    输出：D阵，m*2,m为第state_i行中的非0元的个数，2，第一个元素是这个非零元所在的列，第二个为原来的数字
    '''
    T1=T[state_i,:].reshape([1,T.shape[1]])
    m=T.shape[1]
    D=[]
    for i in range(m):
        if T1[0,i]==0:
            continue
        else:
            D.append([i,T1[0,i]])
    D=np.array(D)
    #为了让我们的D观差更加直观，我们设定一个阵Z,实现按照从小到大排序状态,很难实现，这里就不实现了
    #Z=np.zeros([D.shape[0],D.shape[1]])
    #d=D[:,1]
    #d.sorted(axis=0)
    
    return D
#D=findprob(S1,state_i=3)
            
    
    


#
#这里是对于数据的提取，来生成数据，并且对于数据进行分类，最后得到两组数据
#A=[600000,600010,600015,600016,600019,600028,600030,600031,600036,600048,600050,600104,600111,600123,600256,600348,600362,600383,600489,600518,600519,600547,600549,600585,600837,600887,600999,601006,601088,601166,601169,601288,601299,601318,601328,601336,601398,601601,601628,601668,601669,601688,601699,601766,601788,601818,601857,601899,601901,601989]

def generatedata(A,lx='price',start_date='2015-10-01',end_date='',freq='30min',asset='E',ty=1,ratio=0.3):
    '''
    A:输入的代码序列，里面结构是A=[600000,600010]
    asset='E'
    ty=1代表按照股票划分ratio为界
    ty=2代表按照时间划分ratio为界
    hb代表着所有获得的数据的列表，这里只在TY=1的时候输出，TY=0的时候没有hb
    #输入lx,起始结束时间，股票代码，以及资产类型和数据划分比例，0.3代表7:3，ty表示数据分割类型，ty=1代表依据股票代码分割，ty=2代表依据交易时间分割
    #这里的输出X1，X2分别代表数据分割的前半部分和后半部分，而hb代表着所有获得的数据的列表
    另外注意，数据提取的时候，有的数据会提取失败，导致每一的数据的结果不一样，不过这里不影响试验结果。
    '''
    C=[]
    m=len(A)
    for i in range(len(A)):
        C.append(str(A[i]))

        
    if ty==1:#这里就是对于不同的股票进行分开数据
        hb=[]
        for i in range(m):            
            try:
                df_day,numprice,df=data1(code=C[i],asset=asset,start_date=start_date,end_date=end_date,freq=freq)
                price=data2(df_day,lx,numprice)#这里就把样本数据变成了矩阵，这里我们需要将其合并
                hb.append(price)#这里将所有的都保存在了hb合并List中，接下来我们需要将List中所有矩阵合并
            except:
                    continue
        #return C,hb
    
#df_day,numprice,df=data1(code=m[1],asset='E',start_date='2017-10-01',end_date='',freq='30min')
#m,hb=generatedata(A[:10],lx='price',start_date='2015-10-01',end_date='',freq='30min',asset='E',ty=1,ratio=0.3)
        #
        mhb=len(hb)#这里是所有得到数据的总量
        m1=int(np.floor(mhb*(1-ratio)))#按照比例取前面的作为比照组
        X1=hb[0]       
        for s in range(1,m1):
            X1=np.append(X1,hb[s],axis=0)#将矩阵合并得到第一个
        
        X2=hb[m1]    #这里的风险是这里执行不了，但是一般保证
        
        for s in range(m1+1,mhb):
            X2=np.append(X2,hb[s],axis=0)
        return hb,X1,X2
    #------------------------------------
    elif ty==2:#这个是基于时间对于股票进行分开
        hb1=[]
        hb2=[]
        for i in range(m):
            try:
                df_day,numprice,df=data1(code=C[i],asset=asset,start_date=start_date,end_date=end_date,freq=freq)
                price=data2(df_day,lx,numprice)#这里就把样本数据变成了矩阵，这里我们需要将其合并
                m=len(price)
                m1=int(np.ceil(m*(1-ratio)))
                hb1.append(price[:m1,:])#这里将所有的都保存在了hb合并List中，接下来我们需要将List中所有矩阵合并
                hb2.append(price[m1:,:])
            except:
                    continue
        X1=hb1[0]
        for s in range(1,len(hb1),1):
            X1=np.append(X1,hb1[s],axis=0)
        X2=hb2[0]
        #try:
        for s in range(1,len(hb2)):
            X2=np.append(X2,hb2[s],axis=0)
        #except:
           # X2=hb2[0]
        c=0#这里的c没有任何含义，就是为了保证输出三个元素
        return c,X1,X2
          
    

#

def gendatedata(x1,x2,hb,random_state1=200,n_clusters=49,num_price=8,ratio=0.3):
    '''
    这里的目的是实现对于日线的分类以及实现日线的PQ的计算，
    我们这里的输入有X1，X2，这是按照ty=1生成的不同股票的组合：这里如果想考虑不同时间的不同的转移阵，只用在数据生成的时候，修改一下时间
    hb是一个list，依次为每一只股票的所有数据。这里我们会吧X1，X2合并，合并了找到这里的中心，然后我们基于所有样本hb,去生成各个样本的日线转移的
    矩阵，求前（1-raio)股票的均值，和后ratio股票的均值，实现数据的处理
    
    
    这里我们的输出
    label1是所有股票的日线分类的顺序
    q,s,p是对应的三种阵的列表
    p2,这个是前（1-ratio)比例的数据的转移阵的均值
    p22,这个是后ratio比例的数据的转移阵的均值
    k_center_data#是这对于所有数据的整体进行分类，找到中心
    '''
    X=np.append(x1,x2,axis=0)#合并得到所有的样本
    from sklearn.cluster import KMeans
    kmeans1=KMeans(n_clusters=n_clusters,random_state=random_state1).fit(X)#所有样本的拟合成结果
    k_center_data=kmeans1.cluster_centers_
   # c1=kmeans1.labels_#这里包含具体的类
    label1=[]
    for i in range(len(hb)):
        label=kmeans1.predict(hb[i])
        label1.append(label)
        


    t=[]
    q=[]
    j=[]
    p=[]
    s=[]
    for i in range(len(label1)):
        c=label1[i].reshape([len(label1[i]),1])
        T=generatepair(c,num_price=num_price)#这里直接生成出来我们想要的对
        t.append(T)
        
        Q,J,P,S=compute(T,n_state=n_clusters,lx=2)#z这里构成了所有的阵，将阵装入List，最后再求均值，对比结果。
        q.append(Q)
        j.append(J)
        p.append(P)
        s.append(S)
    #上面已经实现了单独对于每一个股票的日线转移阵的计算，接下来我们考虑计算均值，
    #这里为了方便起见，我们只用s来计算
    m=len(p)
    ms=int(np.floor(m*(1-ratio)))
    
    p1=p[1]
    for i in range(1,ms):
        p1+=p[i]
    p2=p1/ms#这里存在一个问题，由于一些行的分类，别的类没有，因此会直接的导致最后该行求和不是1
    p11=p[ms]
    for i in range(ms+1,m):
        p11+=p[i]
    p22=p11/(m-ms)
    for i in [p2,p22]:
        for s in range(len(i)):
            if sum(i[s])!=0:
                i[s]=i[s]*(1/sum(i[s]))


    return label1,q,s,p,p2,p22,k_center_data




def Logisticdata(num_price,T):
    '''
    目标：实现对于输入为分类数据的处理，我们这里给定的数据是一个m*num_price的类别的表格
    输出的是多组分别的预测得分的均值（打分基于5折交叉验证集的平均值），依次为第2到num_price的样本的预测的正确率情况
    T将所有样本都汇总在一起，m*num_price，m是样本数量，num_price是1天/周期
    输出的X,Y为训练的特征与labels（这里是多元的分类）
    '''
    l=int(T.shape[1]/num_price)
    
    Score=[]
    for i in range(1,num_price):
        s=i
        #print(s)
        X=T[:,:s]
        Y=T[:,i]
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelBinarizer
        from sklearn.model_selection import cross_val_score
        #Y1=LabelBinarizer().fit_transform(Y)
        clf=LogisticRegression(multi_class=multi_class)#.fit(X, Y)
        #clf.fit(X,Y[:])
        scores=cross_val_score(clf,X,Y,cv=4)
        score=np.array(scores).mean()
        Score.append(score)
    return X,Y,Score















    
        

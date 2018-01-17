# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:16:04 2017

@author: Administrator
"""
#想法是将分解的IMF分为三类，分别进行预测，高频用KNN，低频用sin函数拟合，
#剩余用多项式拟合，现在的主要问题是sin函数拟合时需要输入初始参数，而且是求局部极大值还是局部极小值
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as ss
from EMD import EMD
from scipy.optimize import curve_fit
import math
import datetime
from scipy.spatial.distance import pdist
import itertools

#---------------------------------------------------------------------------------
#KNN预测算法主程序
def KNN_pred(data,k,l):
    n=len(data)
    redata=[]
    for i in range(1,n-l+2):
        d=data[i-1:i+l-1]
        redata.append(d)
    redata=np.array(redata)
    newdata=np.zeros((n-l+1,len(redata[0,:])-1))
    for i in range(0,n-l+1):
        for j in range(len(redata[0,:])-1):
            if (redata[i,j+1]-redata[i,j])/redata[i,j]<=0.006:
                newdata[i,j]=0
            elif (redata[i,j+1]-redata[i,j])/redata[i,j]>0.006:
                newdata[i,j]=1
            else:
                newdata[i,j]=-1
    dis=[]
    nn=len(newdata)
    for j in range(1,nn-l+1):
        X=np.vstack([newdata[:][j-1],newdata[:][nn-l]])
        dis.append(pdist(X))
    dis=list(itertools.chain.from_iterable(dis))
    dis=np.array(dis)
    loc=np.argsort(dis)
    loc_k=loc[0:k]
    sum_dis=0
    y_pred=[]
    for h in range(0,k):
        sum_dis+=float(redata[loc_k[h]+1][-1])-float(redata[loc_k[h]][-1])
        #erro+=(k-h)*(float(redata[loc_k[h]+1][-1])-float(redata[loc_k[h]][-1]))/k 
    erro=sum_dis/k
    y_pred=erro+data[-1]
    return y_pred
#-------------------------------------------------------------------------------------
#找到最佳的k和l
def find_best_k_l(train_data,test_data,maxk,maxl):
    error=[]  #误差
    pred=[]
    for k in range(2,maxk):
        for l in range(2,maxl):
            y_pred=KNN_pred(train_data,k,l)
            y_pred=np.array(y_pred)
            error.append(test_data-y_pred)
            pred.append(y_pred)
    error=np.array(error)
    pred=np.array(pred)
    error=abs(error)
    errormat=error.reshape((maxk-2,maxl-2))
    predmat=pred.reshape((maxk-2,maxl-2))
    k,l=np.where(errormat==np.min(errormat))
    return errormat,predmat
#------------------------------------------------------------------------------------
#打开文件
def openfile(location):
    df=pd.read_excel(location)
    df = df.drop(df.loc[np.isnan(df.收盘价.values)].index)
    data=df[['交易时间','收盘价']]
    return data
#---------------------------------------------------------------------------------------
#生成训练集和测试集
def produce_train_test_data(data,n):
    data.columns=['time','return']
    #today = datetime.date.today()
    d=data['time'].tail(1).tolist()
    date_time = datetime.datetime.strptime(str(d[0]),'%Y-%m-%d %H:%M:%S')
    date_time=datetime.date(date_time.year,date_time.month,date_time.day)
    past= date_time - datetime.timedelta(days=n*365)
    thred=datetime.date(2017,9,5)
    if past not in data['time']:
        past=past-datetime.timedelta(days=2)
    da=data.loc[int(data[data['time']==past].index.values):]
    train=data.loc[int(data[data['time']==past].index.values):int(data[data['time']==thred].index.values)-1]
    test=data.loc[int(data[data['time']==thred].index.values):]
    return da,train,test
#-------------------------------------------------------------------------------------
#将IMFs划分为三类，r是趋势项，IMF1是高频，IMF2是低频
def Fine_to_coarse(da,IMFs):
    r=IMFs[-1]
    da=np.zeros((len(IMFs)-1,len(da)))
    t_test=[]
    for i in range(0,len(IMFs)-1):
        da[i,:]=sum(IMFs[0:i+1])
        t_test.append(ss.ttest_1samp(da[i],0))
    me=[]
    for j in range(len(t_test)):
        me.append(t_test[j].pvalue)
    me=np.array(me)
    a=np.where(me<0.05)[0][0]
    IMF1=sum(IMFs[0:a])
    IMF2=sum(IMFs[a:len(IMFs)-1])
    return r,IMF1,IMF2
#----------------------------------------------------------------------------------------
#r是趋势项，deg是多项式拟合的自由度，step是预测的步长    
def r_pred(r,deg,step):
    x = np.arange(1, len(r)+1, 1)
    z1 = np.polyfit(x,r,deg)
    # 生成多项式对象
    p1 = np.poly1d(z1)
#    plt.plot(p1(x),label='fitting values')
#    plt.plot(r,label='residues')
    x1=np.arange(x.max(), x.max()+step, 1)
    r_pre=p1(x1)
    return r_pre
#----------------------------------------------------------------------------------------
#找到IMF2最近的sin模式数据,寻找所有的极小值
def find_nearest_small(data):
    small=[]
    for k in range(1,len(data)-1):
        if data[k]<data[k-1] and data[k]<data[k+1]:
            small.append(k)
    last_small=max(small)
    return last_small
#---------------------------------------------------------------------------------------
#寻找最近的极大值
def find_nearest_big(data):
    big=[]
    for k in range(1,len(data)-1):
        if data[k]>data[k-1] and data[k]>data[k+1]:
            big.append(k)
    last_big=max(big)
    return last_big
#---------------------------------------------------------------------------------------
#预测IMF2的值
def IMF2_pred(IMF2,step,small,big,last):
    L=np.arange(1, len(IMF2)-last+1, 1)
    P=IMF2[last:]
    def fitSine(x, A, f, phi, c):
         return A*np.cos(2*math.pi*f*x+phi) + c
    def initial_parameter(P,small,big,last):
        A=abs(IMF2[big]-IMF2[small])/2
        f=np.pi/(A)
        if big==last:
            phi=0
            c=IMF2[last]-A
        else:
            phi=math.pi
            c=IMF2[last]+A
        return A,f,phi,c
    A,f,phi,c=initial_parameter(P,small,big,last)
    fit_opt, fit_cov = curve_fit(fitSine, L, P,p0=[A,f,phi,c])
    # 提取拟合数据
    coe_A = fit_opt[0]
    coe_f = fit_opt[1]
    coe_phi = fit_opt[2]
    coe_c = fit_opt[3]
    #fitted_P = fitSine(L, coe_A, coe_f, coe_phi, coe_c)
#    plt.plot(P)
#    plt.plot(fitted_P)
    xx=np.arange(len(IMF2)-last+1,len(IMF2)-last+step+1, 1)
    pre_IMF2 = fitSine(xx, coe_A, coe_f, coe_phi, coe_c)
    return pre_IMF2
#-----------------------------------------------------------------------------------
#训练查找最佳的k和l
def find_best_parameter(traindata,num,maxk,maxl):
    bestkk=[]
    bestll=[]
    error=0
    for j in range(num):
        train=traindata[:-num+j]
        test=traindata[-num+j]
        errormat,predmat=find_best_k_l(train,test,maxk,maxl)
        error=error+errormat
    kk,ll=np.where(error==np.min(error))
    bestkk=max(kk)+2
    bestll=max(ll)+2
    print(bestkk,bestll)
    return bestkk,bestll
#-------------------------------------------------------------------------------------
#根据训练结果最佳k和l，预测2017年的收盘价，观察预测结果   
def knn_pred(traindata1,testdata1,prednum,maxk,maxl,num):
    pred=np.zeros(prednum)
    #yy_pred=0
    #new=[]
    error=[]
    bestkk,bestll=find_best_parameter(traindata1,num,maxk,maxl)
    for jj in range(prednum): 
        y_pred=KNN_pred(traindata1,bestkk,bestll)
        y_pred=np.array(y_pred)
        print(y_pred)
        pred[jj]=y_pred
        traindata1=np.hstack((traindata1,testdata1[jj]))
        error.append(pred-testdata1[jj])
        #da=np.hstack((da,y_pred))
    return pred
#---------------------------------------------------------------------------------
#预测结果和真实数据是涨还是跌
def rise_or_fall(data):
    #data=data.tolist()
    diff_data=np.diff(data)
    result=np.sign(diff_data)
    return result

#-------------------------------------------------------------------------------
#评估预测结果，精确度和召回率
def recall_rate(data1,data2):
    result=[]
    label={'11':'TP','1-1':'FN','-11':'FP','-1-1':'TN'}
    for i in range(len(data1)):
        state=str(int(data1[i]))+str(int(data2[i]))
        result.append(label[state])
    count_tp=result.count('TP')    
    count_fn=result.count('FN') 
    count_fp=result.count('FP') 
    #count_tn=result.count('TN') 
    F1=count_tp/(count_tp+count_fp)
    F2=count_tp/(count_tp+count_fn)
    BEP=2*F1*F2/(F1+F2)
    return BEP

#-----------------------------------------------------------------------------------
#主程序
#def main():
data=openfile('data/上证.xls')
t=3
da,traindata,testdata=produce_train_test_data(data,t)
da=np.array(da['return'])
da=np.array(da)
testdata=np.array(testdata['return'])
testdata=np.array(testdata)
traindata=np.array(traindata['return'])
traindata=np.array(traindata)
deg=4
step=1
maxk=22
maxl=10
prednum=len(testdata)
num=prednum
emd=EMD()
emd.extrema_detection = "parabol"
IMFs = emd.emd(da)
r,IMF1,IMF2=Fine_to_coarse(da,IMFs)
#plt.plot(da,label='original')
#plt.plot(IMF1,label='high frequency')
#plt.plot(IMF2,label='low frequency')
#plt.plot(r,label='residue')
#plt.ylabel('Adj Close')
#plt.legend(loc='best')
r_pre=[]
IMF2_pre=[]
traindata_r=r[:-prednum]
testdata_r=r[-prednum:]
traindata2=IMF2[:-prednum]
testdata2=IMF2[-prednum:]
#small=find_nearest_small(traindata2)
#big=find_nearest_big(traindata2)
#last=max(small,big)
#r_pre=r_pred(traindata_r,deg,step)
#IMF2_pre=IMF2_pred(traindata2,step,last)
for j in range(prednum):
    r_pre.append(r_pred(traindata_r,deg,step))
    traindata_r=np.hstack((traindata_r,testdata_r[j]))
    small=find_nearest_small(traindata2)
    big=find_nearest_big(traindata2)
    last=max(small,big)
    IMF2_pre.append(IMF2_pred(traindata2,step,small,big,last))
    traindata2=np.hstack((traindata2,testdata2[j]))
#'''b=np.array(IMF2_pre)
#plt.plot(testdata2,label='true')
#plt.plot(b,label='predict')
#plt.legend(loc='best')
#plt.title('low frequency')
#b=np.array(r_pre)
#plt.plot(testdata_r,label='true')
#plt.plot(b,label='predict')
#plt.legend(loc='best')
#plt.title('residue')'''
testdata1=IMF1[-prednum:]
traindata1=IMF1[:-prednum]
IMF1_pre=knn_pred(traindata1,testdata1,prednum,maxk,maxl,num)
#b=np.array(IMF1_pre)
#plt.plot(testdata1,label='true')
#plt.plot(b,label='predict')
#plt.legend(loc='best')
#plt.title('high frequency')
    #print(r_pre,IMF2_pre,IMF1_pre)
    #return r_pre,IMF2_pre,IMF1_pre
#if __name__=='__main__':
#    main()
IMF1_pre=np.array(IMF1_pre)
IMF2_pre=np.array(IMF2_pre)
r_pre=np.array(r_pre)
pred=IMF1_pre+r_pre[:,0]+IMF2_pre[:,0]
pre_result=rise_or_fall(pred)
true_result=rise_or_fall(testdata)
result=recall_rate(true_result,pre_result)
plt.plot(pred,label='prediction')
plt.plot(testdata,label='true')
plt.legend(loc='best')
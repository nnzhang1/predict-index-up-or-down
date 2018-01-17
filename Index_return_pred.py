# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 14:21:53 2017

@author: ningningzhang
"""

#import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy.spatial.distance import pdist
import itertools
from sklearn import preprocessing
#from sklearn.linear_model import LinearRegression
#---------------------------------------------------------------------------------------
#KNN预测算法主程序
def KNN_pred(data,k,l,a,b):
    n=len(data)
    redata=[]
    for i in range(1,n-l+2):
        redata.append(data[i-1:i+l-1])
    redata=np.array(redata)
    #redata1=np.diff(redata,axis=1)
    newdata=np.sign(redata)
    dis_sym=[]
    nn=len(newdata)
    for j in range(0,nn-l):
        X=np.vstack([newdata[:][j],newdata[:][nn-l]])
        dis_sym.append(pdist(X))
    dis_sym=list(itertools.chain.from_iterable(dis_sym))
    dis_sym=np.array(dis_sym)
    dis=[]
    for ii in range(0,nn-l):
        X=np.vstack([redata[:][ii],redata[:][nn-l]])
        dis.append(pdist(X))
    dis=list(itertools.chain.from_iterable(dis))
    dis=np.array(dis)
    sumdis=a*preprocessing.scale(dis_sym)+b*preprocessing.scale(dis)
    loc=np.argsort(sumdis)
    loc_k=loc[0:k]
    sum_dis=0
    y_pred=[]
    for h in range(0,k):
        sum_dis+=float(redata[loc_k[h]][0])-float(redata[loc_k[h]-1][-1])
    erro=sum_dis/k   
    y_pred=erro+data[-1]
    return y_pred
#-------------------------------------------------------------------------------------
#找到最佳的k和l
def find_best_k_l(train_data,test_data,maxk,maxl,a,b):
    error=[]  #误差
    pred=[]
    for k in range(3,maxk):
        for l in range(3,maxl):
            y_pred=KNN_pred(train_data,k,l,a,b)
            y_pred=np.array(y_pred)
            error.append(test_data-y_pred)
            pred.append(y_pred)
    error=np.array(error)
    pred=np.array(pred)
    error=abs(error)
    errormat=error.reshape((maxk-3,maxl-3))
    predmat=pred.reshape((maxk-3,maxl-3))
    k,l=np.where(errormat==np.min(errormat))
    return errormat,predmat
#------------------------------------------------------------------------------------
#打开文件
def openfile(location):
    df=pd.read_excel(location)
    df = df.drop(df.loc[np.isnan(df.收盘价.values)].index)
    data=df[['交易时间','涨跌幅%','成交量']]
    return data
#--------------------------------------------------------------------------------------
#生成训练集和测试集
def produce_train_test_data(data,n):
    data.columns=['time','return','volume']
    #today = datetime.date.today()
    d=data['time'].tail(1).tolist()
    date_time = datetime.datetime.strptime(str(d[0]),'%Y-%m-%d %H:%M:%S')
    date_time=datetime.date(date_time.year,date_time.month,date_time.day)
    past= date_time - datetime.timedelta(days=n*365)
    thred=datetime.date(2017,1,3)
    if past not in data['time']:
        past=past-datetime.timedelta(days=2)
    da=data.loc[int(data[data['time']==past].index.values):]
    train=data.loc[int(data[data['time']==past].index.values):int(data[data['time']==thred].index.values)-1]
    test=data.loc[int(data[data['time']==thred].index.values):]
    return da,train,test
#--------------------------------------------------------------------------------------
#训练查找最佳的k和l
def find_best_parameter(data,num,maxk,maxl):
    bestkk=[]
    bestll=[]
    merr=[]
    all=np.linspace(0.5,1,5)
    for a in all:
        b=1-a
        error=0
        for j in range(num):
            train=data[:-num+j]
            test=data[-num+j]
            errormat,predmat=find_best_k_l(train,test,maxk,maxl,a,b)
            error=error+errormat
        train_true=data[-num:]
        kk,ll=np.where(error==np.min(error))
        bestkk.append(max(kk)+3)
        bestll.append(max(ll)+3)
        pred=[]
        for j in range(num):
            train=data[:-num+j]
            pred.append(KNN_pred(train,max(kk)+3,max(ll)+3,a,b)) 
        pred=np.array(pred)
        err=train_true-pred
        merr.append(np.mean(err))
    loc=np.where(merr==np.min(merr))
    loc=np.array(loc)
    loc=loc.tolist()
    loc=np.max(loc,axis=1)
    bestkkk=bestkk[loc]
    bestlll=bestll[loc]
    besta=all[loc]
    bestb=1-besta
    return bestkkk,bestlll,besta,bestb
#-------------------------------------------------------------------------------------
#根据训练结果最佳k和l，预测2017年的收盘价，观察预测结果   
def knn_pred(data,pp,prednum):
    error=[]
    pred=np.zeros(prednum)
    for jj in range(prednum):
        bestkk,bestll,a,b=find_best_parameter(data,num,maxk,maxl)
        y_pred=KNN_pred(data,bestkk,bestll,a,b)
        y_pred=np.array(y_pred)
        pred[jj]=y_pred
        data=np.hstack((data,pp[jj]))
        error.append(y_pred-pp[jj])
    return data,pred,error
#---------------------------------------------------------------------------------
#预测结果和真实数据是涨还是跌
def rise_or_fall(data):
    data=data.tolist()
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
    #count_tn=result.count('TN') 
    F1=count_tp/(count_tp+count_fp)
    F2=count_tp/(count_tp+count_fn)
    BEP=2*F1*F2/(F1+F2)
    #F3=(count_tp+count_tn)/(count_tp+count_tn+count_fn+count_fp)
    #return BEP,F3
    return BEP

#--------------------------------------------------------------------------------
#主程序
#def main():
data=openfile('data/上证.xls')
t=3
da,traindata,testdata=produce_train_test_data(data,t)
p=da['return']
pp=testdata['return']
pp=np.array(pp)
num=22
prednum=len(testdata)
maxk=30
maxl=30
oridata=traindata['return']
oridata=oridata.tolist()
oridata=np.array(oridata)
da,pred,error=knn_pred(oridata,pp,prednum)
truedata=p[-prednum:]
truedata=truedata.reset_index(drop=True)
plt.plot(truedata,label='true values')
plt.plot(pred,label='predicted values')
plt.legend(loc='best')
plt.savefig('预测图.png')
truedata_result=rise_or_fall(truedata)
pred_result=rise_or_fall(pred)
result= recall_rate(truedata_result,pred_result)
print(result)
#return result
#if __name__=='__main__':
#    main()
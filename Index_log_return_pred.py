# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:23:47 2017

@author: ningningzhang
"""
import math
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import datetime
from scipy.spatial.distance import pdist
import itertools
#---------------------------------------------------------------------------------------
#KNN预测算法主程序
def KNN_pred(data,k,l,h):
    n=len(data)
    redata=[]
    for i in range(1,n-l+2):
        redata.append(data[i-1:i+l-1])
    redata=np.array(redata)
    redata1=pd.DataFrame(redata)
    redata1=redata1.diff(axis=1)
    redata1=np.array(redata1)
    redata1=redata1[:,1:]
    newdata=np.zeros([len(redata1),len(redata1[0])])
    for ii in range(len(redata1)):
        for jj in range(len(redata1[0])):
            if redata1[ii][jj]==0:
                newdata[ii][jj]=0
            elif redata1[ii][jj]>0:
                newdata[ii][jj]=1
            else:
                newdata[ii][jj]=-1
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
        sum_dis+=float(redata[loc_k[h]][0])-float(redata[loc_k[h]-1][-1])
    erro=sum_dis/k   
    y_pred=erro+data[-1]
    return y_pred
#-------------------------------------------------------------------------------------
#找到最佳的k和l
def find_best_k_l(train_data,test_data,maxk,maxl,n):
    error=[]  #误差
    pred=[]
    for k in range(2,maxk):
        for l in range(2,maxl):
            y_pred=KNN_pred(train_data,k,l,n)
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
    #thred=datetime.date(2014,1,2)
    #redata=data.loc[int(data[data['交易时间']==thred].index.values):]
    return data
#--------------------------------------------------------------------------------------
#生成训练集和测试集
def produce_train_test_data(data,n):
    data.columns=['time','values']
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
#对数收益率
def rate_log_return(data):
    p=np.zeros(len(data)-1)
    for i in range(len(data)-1):
        p[i]=math.log(data[i+1]/data[i])
    return p
#------------------------------------------------------------------------------------------
#训练查找最佳的k和l
def find_best_parameter(data,num,maxk,maxl):
    error=0
    for j in range(num):
        train=data[:-num+j]
        test=data[-num+j]
        errormat,predmat=find_best_k_l(train,test,maxk,maxl,1)
        error=error+errormat
    kk,ll=np.where(error==np.min(error))
    bestkk=max(kk)+2
    bestll=max(ll)+2
    return bestkk,bestll
#-------------------------------------------------------------------------------------
#根据训练结果最佳k和l，预测2017年的收盘价，观察预测结果   
def knn_pred(data,pp,prednum,bestkk,bestll):
    pred=np.zeros(prednum)
    #yy_pred=0
    #new=[]
    da=data
    error=[]
    for jj in range(prednum):
        y_pred=KNN_pred(da,bestkk,bestll,1)
        y_pred=np.array(y_pred)
        pred[jj]=y_pred
        da=np.hstack((da,pp[jj]))
        error.append(pred-pp[jj])
        #da=np.hstack((da,y_pred))
    return da,pred,error
#---------------------------------------------------------------------------------
#预测结果和真实数据是涨还是跌
def rise_or_fall(diff_data):
    #data=data.tolist()
    #diff_data=np.diff(data)
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

#--------------------------------------------------------------------------------
#主程序
#def main():
data=openfile('data/上证.xls')
t=3
da,traindata,testdata=produce_train_test_data(data,t)
p=da['values']
p=np.array(p)
p1=rate_log_return(p)
pp=testdata['values']
pp=np.array(pp)
pp1=rate_log_return(pp)
pp1=np.array(pp1)
#plt.plot_date(da['交易时间'],p)
num=22
prednum=len(testdata)-1
maxk=30
maxl=30    
oridata=np.array(traindata['values'])
oridata1=rate_log_return(oridata)
bestkk,bestll=find_best_parameter(oridata1,num,maxk,maxl)
da,pred,error=knn_pred(oridata1,pp1,prednum,bestkk,bestll)
truedata=p1[-prednum:]
#truedata=truedata.reset_index(drop=True)
#plt.plot(truedata,label='true values')
#plt.plot(pred,label='predicted values')
#plt.legend(loc='best')
truedata_result=rise_or_fall(truedata)
#truedata_result.to_excel('truedata.xls')
pred_result=rise_or_fall(pred)
result= recall_rate(truedata_result,pred_result)
print(result)
print(truedata)
print(pred)
#return result
#if __name__=='__main__':
#    main()
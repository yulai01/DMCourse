#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
def prediction(df,userdf,Nn=15):#Nn邻居个数
    corr=df.T.corr();
    rats=userdf.copy()
    for usrid in userdf.index:
        dfnull=df.loc[usrid][df.loc[usrid].isnull()]
        usrv=df.loc[usrid].mean()#评价平均值
        for i in range(len(dfnull)):
            nft=(df[dfnull.index[i]]).notnull()
            #获取邻居列表
            if(Nn<=len(nft)):
                nlist=df[dfnull.index[i]][nft][:Nn]
            else:
                nlist=df[dfnull.index[i]][nft][:len(nft)]
            nlist=nlist[corr.loc[usrid,nlist.index].notnull()]
            nratsum=0
            corsum=0
            if(0!=nlist.size):
                nv=df.loc[nlist.index,:].T.mean()#邻居评价平均值
                for index in nlist.index:
                    ncor=corr.loc[usrid,index]
                    nratsum+=ncor*(df[dfnull.index[i]][index]-nv[index])
                    corsum+=abs(ncor)
                if(corsum!=0):
                    rats.at[usrid,dfnull.index[i]]= usrv + nratsum/corsum
                else:
                    rats.at[usrid,dfnull.index[i]]= usrv
            else:
                rats.at[usrid,dfnull.index[i]]= None
    return rats
def recomm(df,userdf,Nn=15,TopN=3):
    ratings=prediction(df,userdf,Nn)#获取预测评分
    recomm=[]#存放推荐结果
    for usrid in userdf.index:
        #获取按NA值获取未评分项
        ratft=userdf.loc[usrid].isnull()
        ratnull=ratings.loc[usrid][ratft]
        #对预测评分进行排序
        if(len(ratnull)>=TopN):
            sortlist=(ratnull.sort_values(ascending=False)).index[:TopN]
        else:
            sortlist=ratnull.sort_values(ascending=False).index[:len(ratnull)]
        recomm.append(sortlist)
    return ratings,recomm


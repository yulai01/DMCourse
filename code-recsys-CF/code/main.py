#-*- coding: utf-8 -*-
#使用基于UBCF算法对电影进行推荐
from __future__ import print_function
import pandas as pd
from recommender import recomm 
############    主程序   ##############
if __name__ == "__main__":
    print("\n--------------使用基于UBCF算法对电影进行推荐 运行中... -----------\n")
    traindata = pd.read_csv('../data/u1.base',sep='\t', header=None,index_col=None)
    testdata = pd.read_csv('../data/u2.test',sep='\t', header=None,index_col=None)
    #删除时间标签列
    traindata.drop(3,axis=1, inplace=True)
    testdata.drop(3,axis=1, inplace=True)
    #行与列重新命名
    traindata.rename(columns={0:'userid',1:'movid',2:'rat'}, inplace=True)
    testdata.rename(columns={0:'userid',1:'movid',2:'rat'}, inplace=True)
    traindf=traindata.pivot(index='userid', columns='movid', values='rat')
    testdf=testdata.pivot(index='userid', columns='movid', values='rat')
    traindf.rename(index={i:'usr%d'%(i) for i in traindf.index} , inplace=True)
    traindf.rename(columns={i:'mov%d'%(i) for i in traindf.columns} , inplace=True)
    testdf.rename(index={i:'usr%d'%(i) for i in testdf.index} , inplace=True)
    testdf.rename(columns={i:'mov%d'%(i) for i in testdf.columns} , inplace=True)
    userdf=traindf.loc[testdf.index]
    #获取预测评分和推荐列表
    trainrats,trainrecomm=recomm(traindf,userdf)
# -*- coding:utf-8 -*-
# 使用ID3算法进行分类
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz

data = pd.read_csv('../data/titanic_data.csv', encoding='utf-8')
data.drop(['PassengerId'], axis=1, inplace=True)    # 舍弃ID列，不适合作为特征

# 数据是类别标签，将其转换为数，用1表示男，0表示女。
data.loc[data['Sex'] == 'male', 'Sex'] = 1
data.loc[data['Sex'] == 'female', 'Sex'] = 0
data.fillna(int(data.Age.mean()), inplace=True)
print(data.head(5))   # 查看数据

X = data.iloc[:, 1:3]    # 为便于展示，未考虑年龄（最后一列）
y = data.iloc[:, 0]

dtc = DTC(criterion='entropy')    # 初始化决策树对象，基于信息熵
dtc.fit(X, y)    # 训练模型
print('输出准确率：', dtc.score(X,y))

# 可视化决策树，导出结果是一个dot文件，需要安装Graphviz才能转换为.pdf或.png格式
with open('../tmp/tree.dot', 'w') as f:
    f = export_graphviz(dtc, feature_names=X.columns, out_file=f)

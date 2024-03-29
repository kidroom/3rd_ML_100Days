# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

data_path = 'E:/data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()

#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
train_num = train_Y.shape[0]
df.head()

# 空值補 0, 做羅吉斯迴歸
df_0 = df.fillna(0)
train_X = df_0[:train_num]
estimator = LogisticRegression(solver='liblinear')
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
#補-1
df_m1 = df.fillna(-1)
train_X = df_m1[:train_num]
estimator = LogisticRegression(solver='liblinear')
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
#補平均值
df_mn = df.fillna(df.mean())
train_X = df_mn[:train_num]
estimator = LogisticRegression(solver='liblinear')
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

#標準化
df_0_st = StandardScaler().fit_transform(df_0)
train_X = df_0_st[:train_num]
estimator = LogisticRegression(solver='liblinear')
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

df_m1_st = StandardScaler().fit_transform(df_m1)
train_X = df_m1_st[:train_num]
estimator = LogisticRegression(solver='liblinear')
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

df_mn_st = StandardScaler().fit_transform(df_mn)
train_X = df_mn_st[:train_num]
estimator = LogisticRegression(solver='liblinear')
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

#最大化最小化
df_0_mnx = MinMaxScaler().fit_transform(df_0)
train_X = df_0_mnx[:train_num]
estimator = LogisticRegression(solver='liblinear')
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

df_m1_mnx = MinMaxScaler().fit_transform(df_m1)
train_X = df_m1_mnx[:train_num]
estimator = LogisticRegression(solver='liblinear')
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

df_mn_mnx = MinMaxScaler().fit_transform(df_mn)
train_X = df_mn_mnx[:train_num]
estimator = LogisticRegression(solver='liblinear')
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

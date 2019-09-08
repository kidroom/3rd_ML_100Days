import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

data_path = 'E:/3rd_ML_100Days/homework/data/'
df_train = pd.read_csv(data_path + 'house_train.csv.gz')

# 削減文字型欄位, 只剩數值型欄位


train_Y = np.log1p(df_train['SalePrice'])
df = df_train.drop(['Id', 'SalePrice'] , axis=1)
plt.hist(df['1stFlrSF'],bins=1000)
plt.show()
print(df['1stFlrSF'].mean())

num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

df = df[num_features]
df = df.fillna(-1)
MMEncoder = MinMaxScaler()
print(df['1stFlrSF'])

keep_indexs = (df['1stFlrSF']> 450) & (df['1stFlrSF']< 2250)
dfr = df_train[keep_indexs]
train_Y = train_Y[keep_indexs]
sns.regplot(x = dfr['1stFlrSF'], y=train_Y)
plt.show()

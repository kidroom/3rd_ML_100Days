# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
import time

# 設定 data_path
dir_data = 'E:/data/'

f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()

app_train['AMT_ANNUITY'].describe()

# 1: 計算 AMT_ANNUITY 的 q0 - q100
i=0
q_all = [np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = i) for i in range(100)]

pd.DataFrame({'q': list(range(100)),
              'value': q_all})

print(q_all)



# 2.1 將 NAs 以 q50 填補
print("Before replace NAs, numbers of row that AMT_ANNUITY is NAs: %i" % sum(app_train['AMT_ANNUITY'].isnull()))

q_50 = np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = i)
app_train.loc[app_train['AMT_ANNUITY'].isnull(),'AMT_ANNUITY'] = q_50
print("After replace NAs, numbers of row that AMT_ANNUITY is NAs: %i" % sum(app_train['AMT_ANNUITY'].isnull()))

# 2.2 Normalize values to -1 to 1
print("== Original data range ==")
print(app_train['AMT_ANNUITY'].describe())

app_train['AMT_ANNUITY'].hist(bins = 50)
plt.title("Original")
plt.show()
value = app_train['AMT_ANNUITY'].values

def normalize_value(value):

    app_train['AMT_ANNUITY_Norm1'] = ( value - np.mean(value) ) / ( np.std(value) )
    app_train['AMT_ANNUITY_Norm1'].hist(bins = 50)
    plt.title("Normalized with Z-transform")
    plt.show()

    app_train['AMT_ANNUITY_Norm2'] = ( value - min(value) ) / ( max(value) - min(value) )
    app_train['AMT_ANNUITY_Norm2'].hist(bins = 50)
    plt.title("Normalized to 0 ~ 1")
    plt.show()

    app_train['AMT_ANNUITY_Norm3'] = (( value - min(value) ) / ( max(value) - min(value) ) - 0.5)*2
    app_train['AMT_ANNUITY_Norm3'].hist(bins = 50)
    plt.title("Normalized to -1 ~ 1")
    plt.show()

    return value

app_train['AMT_ANNUITY_NORMALIZED'] = normalize_value(app_train['AMT_ANNUITY'])

print("== Normalized data range ==")
app_train['AMT_ANNUITY_NORMALIZED'].describe()

# 3
print("Before replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: %i" % sum(app_train['AMT_GOODS_PRICE'].isnull()))

# 列出重複最多的數值
value_most = mode(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'])
print(f'眾數 : {value_most}')

mode_goods_price = list(app_train['AMT_GOODS_PRICE'].value_counts().index)
app_train.loc[app_train['AMT_GOODS_PRICE'].isnull(), 'AMT_GOODS_PRICE'] = mode_goods_price[0]

print("After replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: %i" % sum(app_train['AMT_GOODS_PRICE'].isnull()))
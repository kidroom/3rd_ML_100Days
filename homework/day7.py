# 載入基本套件
import pandas as pd
import numpy as np

# 讀取訓練與測試資料
data_path = 'E:/data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')
df_train.shape

# 重組資料成為訓練 / 預測用格式
train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()

# 秀出資料欄位的類型與數量
dtype_df = df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df = dtype_df.groupby("Column Type").aggregate('count').reset_index()
dtype_df

#確定只有 int64, float64, object 三種類型後, 分別將欄位名稱存於三個 list 中
int_features = []
float_features = []
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64':
        float_features.append(feature)
    elif dtype == 'int64':
        int_features.append(feature)
    else:
        object_features.append(feature)


print(f'int average : {df[int_features].mean()}')
print(f'float average : {df[float_features].mean()}')
print(f'object average : {df[object_features].mean()}')

print(f'int max : {df[int_features].max()}')
print(f'float max : {df[float_features].max()}')
print(f'object max : {df[object_features].max()}')

print(f'int nunique : {df[int_features].nunique()}')
print(f'float nunique : {df[float_features].nunique()}')
print(f'object nunique : {df[object_features].nunique()}')

print('homework1 : object在使用平均值時無法計算，因為物件裡面有非數值的資料型態造成無法計算。')
print('homework2 : string, date, bool, short, double, 大多數還是屬於三大類型。其中物件的資料型態最難處理，其裡面可以包含的所有資料型態。')
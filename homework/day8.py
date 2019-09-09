# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 設定 data_path
dir_data = 'E:/data/'

f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)

print(app_train.head())

print(app_train['AMT_INCOME_TOTAL'].mean())
print(np.std(app_train['AMT_INCOME_TOTAL'],ddof=1))


plt.hist(app_train['AMT_INCOME_TOTAL'])
plt.show()


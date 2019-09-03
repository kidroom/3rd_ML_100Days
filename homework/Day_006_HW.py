import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

dir_data = 'E:/data/'
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)

app_train = pd.read_csv(f_app_train)
sub_train = pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START'])

onehotcode = pd.get_dummies(sub_train)
print('before : ' + str(sub_train.shape))
print('after : ' + str(onehotcode.shape))
print('before : ' + str(sub_train.head()))
print('after : ' + str(onehotcode.head()))



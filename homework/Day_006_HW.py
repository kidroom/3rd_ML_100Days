import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
dir_data = 'E:/data/'
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)

sub_train = pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START'])

lblencode = LabelEncoder()
sub_train = lblencode.fit_transform(sub_train[:, 0])
print(sub_train.shape)
print(sub_train)
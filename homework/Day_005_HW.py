import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
import requests

target_url='https://raw.githubusercontent.com/vashineyu/slides_and_others/master/tutorial/examples/imagenet_urls_examples.txt'

dir_data = 'E:/data/'
f_app = os.path.join(dir_data, 'day5_data.txt')
app_train = pd.read_csv(f_app,header = None, sep='\t')
data = pd.DataFrame(data=app_train)
print(data.head())

response = requests.get(target_url)
#urltxt = pd.read_csv(f_app,header = None, sep='\t')
urltxt = response.text

split_tag = '\n'
urltxt = urltxt.split(split_tag)

split_tag = '\t'
for string in urltxt:
    imagelist = string.split(split_tag)
    print(imagelist[1])

import os
import io
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import requests

target_url='https://raw.githubusercontent.com/vashineyu/slides_and_others/master/tutorial/examples/imagenet_urls_examples.txt'

response = requests.get(target_url)
urltxt = response.text

urltxt = urltxt.split('\n')
i=0

name=[]
url=[]

for string in urltxt:
    name.append(string.split('\t')[0])
    try:
        url.append(string.split('\t')[1])
    except:
        url.append("")

data_dict = {'name': name, 'url':url}
data = pd.DataFrame(data_dict)

k=5
i=0
while i < k:
    try:
        img=Image.open(io.BytesIO((requests.get(data.loc[i][1])).content))
    except :
        k+=1
    else:
        plt.imshow(img)
        plt.show()
    i+=1

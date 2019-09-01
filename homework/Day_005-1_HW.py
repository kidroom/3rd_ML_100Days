import pandas as pd
import numpy as np

people = np.random.randint(100000000, size=3)

info = {'國家': ['Taiwan', 'America', 'England'], '人口': people}
df = pd.DataFrame(info)

print(df)
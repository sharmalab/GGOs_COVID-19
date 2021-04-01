import pdb
import numpy as np
import pandas as pd
#data = np.loadtxt('TXT_cloud.txt')
data = pd.read_csv('TXT_cloud.txt', header=None, sep='\s+', dtype=np.float64).values
#pdb.set_trace()
#print(data)
w, h = data.shape
#print(w, h)
for i in range(w):
    for j in range(h):
        if data[i][3] >=100 and data[i][4] >=100 and data[i][5] >=100:
           print(data[i][j]) 

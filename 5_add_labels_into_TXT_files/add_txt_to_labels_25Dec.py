import os
import numpy as np
import pandas as pd
import pdb
txt_file_path = "/home/ms/Desktop/Point_Cloud_Monjoy/data_19Jan2021/5_text_PointCloud/"
labeled_txt_save_path = "/home/ms/Desktop/Point_Cloud_Monjoy/data_19Jan2021/6_labeled_txt/"


single_files = os.listdir(txt_file_path)


if not os.path.exists(labeled_txt_save_path):
    os.makedirs(labeled_txt_save_path)
    print("Created ouput directory: " + labeled_txt_save_path)



for single_file in single_files:
    if single_file.endswith(".txt"):
       fname = os.path.basename(single_file)
       #pdb.set_trace()
       #base, ext = os.path.splitext(fname)
       df= pd.read_csv(txt_file_path+single_file, header=None, sep='\s+', dtype=np.float64)
       df[len(df.columns)] = 0
       #df
       w, h = df.shape
       #w, h
       for i in range(w):
           if df.iloc[i,3] >= 100 and df.iloc[i,4] >= 100 and df.iloc[i,5] >= 100:
              df.iloc[i, 6] = 1
              #df
       df.to_csv(labeled_txt_save_path+fname, sep='\t', index=False, header=False, float_format='%.6f')

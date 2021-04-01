#This code can be used to get convex hull, convex points. Input was binary. Contour color has been set as WHITE(255, 255, 255). Working nicely
#on lung CT images. 

import os
import cv2
import glob
import copy
import shutil
import mahotas
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from PIL import ImageMath
from matplotlib import image
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.image as mpimg
#from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage import measure, morphology
from scipy.spatial import distance as dist
from skimage.measure import label, regionprops
from skimage.morphology import ball, binary_closing
from skimage.morphology import watershed as skwater
from skimage import measure, morphology, segmentation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


import pdb
# Binary Image Path. This Binary Images has been created by running the code "1_original_to_binary.py"
data_path = "/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-02_Results/step_1_original_to_binary/"
save_path = "/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-02_Results/step_1a_convexHull/"

#get list of pngs source files
source_files = os.listdir(data_path)

for file_single in source_files:
    if not os.path.exists(save_path + file_single):
        os.makedirs(save_path + file_single)
        print("Created ouput directory: " + save_path + file_single)
    g = glob(data_path + file_single + "/*.png")
    for image in g:
        src = cv2.imread(image)
        print(image)
        fname = os.path.basename(image)
        #Convert into the gray
        img_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY) # convert to grayscale
        ret, thresh = cv2.threshold(img_gray, 127, 255, 0)    
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
        #print len(contours)
        
        for j in range(len(contours)):
            cnt = contours[j]
            hull = cv2.convexHull(cnt,returnPoints = False)
            defects = cv2.convexityDefects(cnt,hull)
            #print(len(hull), len(defects))
            #pdb.set_trace()
            try:
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    D = dist.euclidean((start[0], start[1]), (end[0], end[1]))
                    #print(D)
                    if D <= 80:
                        cv2.line(src,start,end,[255,255,255],1)
                        #cv2.circle(src,far,1,[0,0,255],-1)
            except AttributeError:
                print('has no attribute')
        ########################### fill the closed regions
        src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        filled = ndimage.binary_fill_holes(src)
        mpimg.imsave(save_path + file_single+'/'+fname, filled, cmap='gray')
        #plt.imshow(filled, cmap='gray')
        #plt.show()




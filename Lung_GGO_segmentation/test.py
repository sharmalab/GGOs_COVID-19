""" Color based K-means"""

import numpy as np
import cv2
import os
import glob
from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
import pdb

heatMap_image_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/check/test/'
save_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/check/only_GGO/'
g= glob(heatMap_image_path + "/*.png")
#
for image in g:
  fname_image = os.path.basename(image)
  img = cv2.imread(image)
  Z = np.float32(img.reshape((-1,3)))
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 4
  _,labels,centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  labels = labels.reshape((img.shape[:-1]))
  reduced = np.uint8(centers)[labels]
  for i, c in enumerate(centers):
    mask = cv2.inRange(labels, i, i)
    mask = np.dstack([mask]*3) # Make it 3 channel
    ex_img = cv2.bitwise_and(img, mask)
    ex_reduced = cv2.bitwise_and(reduced, mask)
    hsv = cv2.cvtColor(ex_reduced, cv2.COLOR_BGR2HSV)
    lower_red = np.array([110,50,50]) 
    upper_red = np.array([130,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    #if cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(reduced, reduced, mask= mask1)
    #print(mask1)
    #if res > 0:
      #plt.imshow(res)
      #plt.show()
    #pdb.set_trace()
    cv2.imwrite(save_path+fname_image, res)


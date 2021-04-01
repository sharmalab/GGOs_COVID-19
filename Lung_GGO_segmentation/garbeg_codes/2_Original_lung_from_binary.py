""" This code can be used to extract Only original lung from original image and binary mask (resulted from step-1 code)"""
#################################################
# 2_Original_lung_from_binary.py for Python 3   #
# extract Only original lung from original      #
#image and binary mask                          s#
#                                               #                       
#     Written by Monjoy Saha                    #
#        monjoybme@gmail.com                    #
#          02 July 2020                         #
#                                               #
#################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import os

import glob
from glob import glob
import pandas as pd

from PIL import Image
import pdb
#skimage image processing packages
from skimage import measure, morphology
from skimage.morphology import ball, binary_closing
from skimage.measure import label, regionprops
import copy
from skimage import measure, morphology, segmentation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage as ndimage
import mahotas

def ShowImage(title,img,ctype):
  plt.figure(figsize=(10, 10))
  if ctype=='bgr':
    b,g,r = cv2.split(img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    plt.imshow(rgb_img)
  elif ctype=='hsv':
    rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    plt.imshow(rgb)
  elif ctype=='gray':
    plt.imshow(img,cmap='gray')
  elif ctype=='rgb':
    plt.imshow(img)
  else:
    raise Exception("Unknown colour type")
  plt.axis('off')
  plt.title(title)
  plt.show()

##############################
original_image_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/CT-03_pngs/'
mask_image_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-03_results/step_1a_convexHull/'
save_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-03_results/step_2_originalMask/'

##############################################################
#get list of png source files
source_files_original = os.listdir(original_image_path)
source_files_mask = os.listdir(mask_image_path)
#pdb.set_trace()

for single_file in source_files_original: 
  for mask in source_files_mask:
    if single_file == mask:
      images = glob(original_image_path + single_file + "/*.png")
      masks= glob(mask_image_path + mask + "/*.png")
      for image in images:
        fname_image_original = os.path.basename(image)
        for msk in masks:
          fname_image_mask = os.path.basename(msk)
          if fname_image_original == fname_image_mask:
            img_read = cv2.imread(image)
            msk_read = cv2.imread(msk)
            result = cv2.subtract(msk_read, img_read)
            result = Image.fromarray((result * 255).astype(np.uint8))
            #pdb.set_trace()
            if not os.path.exists(save_path + single_file):
              os.makedirs(save_path + single_file)
              print("Created ouput directory: " + save_path + single_file)
            result.save(save_path+single_file+'/'+fname_image_original)
      
      
    
    
  
 





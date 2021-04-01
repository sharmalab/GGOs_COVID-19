""" This code can be used to get a contour image. Here contour will be generated based on the binary mask"""
#################################################
# 5_contour_image.py for Python 3               #
#                                               #
#                                               #                       
#     Written by Monjoy Saha                    #
#        monjoybme@gmail.com                    #
#          02 July 2020                         #
#                                               #
#################################################

import os
import cv2
import glob
import numpy as np
from glob import glob
from PIL import Image
from skimage import morphology
import scipy.ndimage as ndimage
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import pdb

original_image_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/CT-01_pngs/'      
binary_image_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-01_results_only_GGO_step-4/'
save_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-01_Final_results_step-5/'
#OrImgs = glob(original_image_path + "/*.png")
#BiImgs = glob(binary_image_path + "/*.png")

#get list of png source files
source_files_original = os.listdir(original_image_path)
source_files_binary = os.listdir(binary_image_path)

for single_file in source_files_original: 
  for mask in source_files_binary:
      if single_file == mask:
          OrImgs = glob(original_image_path + single_file + "/*.png")
          BiImgs= glob(binary_image_path + mask + "/*.png")
          for OrImg in OrImgs:
              fname_image_OrImg = os.path.basename(OrImg)
              for BiImg in BiImgs:
                  fname_image_BiImg = os.path.basename(BiImg)
                  if fname_image_OrImg == fname_image_BiImg:
                      img_OrImg = cv2.imread(OrImg)
                      img_BiImg = cv2.imread(BiImg)
                      img_BiImg = cv2.cvtColor(img_BiImg, cv2.COLOR_BGR2GRAY)
                      # Fill holes within the binary image
                      img_BiImg = ndimage.binary_fill_holes(img_BiImg).astype(np.uint8)
                      contours, hierarchy =  cv2.findContours(img_BiImg.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                      cv2.drawContours(img_OrImg, contours, -1, (255,0,0), 1)
                      mpimg.imsave(save_path+single_file+fname_image_OrImg, img_OrImg, cmap='gray')
            
            



            
  

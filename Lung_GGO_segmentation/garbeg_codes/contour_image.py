""" Color based K-means"""
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

original_image_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/CT-01_pngs/study_0267/'      
binary_image_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-01_results_only_GGO_step-4/study_0267/'
save_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-01_Final_results_step-5/study_0267/'
OrImgs = glob(original_image_path + "/*.png")
BiImgs = glob(binary_image_path + "/*.png")


#
for OrImg in OrImgs:
    fname_image_OrImg = os.path.basename(OrImg)
    for BiImg in BiImgs:
        fname_image_BiImg = os.path.basename(BiImg)
        if fname_image_OrImg == fname_image_BiImg:
            img_OrImg = cv2.imread(OrImg)
            #img_OrImg = cv2.cvtColor(img_OrImg, cv2.COLOR_BGR2GRAY)
            img_BiImg = cv2.imread(BiImg)
            img_BiImg = cv2.cvtColor(img_BiImg, cv2.COLOR_BGR2GRAY)
            # Fill holes within the binary image
            img_BiImg = ndimage.binary_fill_holes(img_BiImg).astype(np.uint8)
            contours, hierarchy =  cv2.findContours(img_BiImg.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(img_OrImg, contours, -1, (0, 0, 255), 3)
            cv2.drawContours(img_OrImg, contours, -1, (255,0,0), 1)
            
            mpimg.imsave(save_path+fname_image_OrImg, img_OrImg, cmap='gray')
            
            



            
  

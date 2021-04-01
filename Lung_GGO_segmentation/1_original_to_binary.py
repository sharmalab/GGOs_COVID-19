"""https://stackoverflow.com/questions/49834264/mri-brain-tumor-image-processing-and-segmentation-skull-removing"""
""" This code can be used to segment main lung from original images into the form of binary images"""
#################################################
# 1_original_to_binary.py for Python 3          #
# Segment main lung from original images        #
#into the form of binary images                 #
#                                               #                       
#     Written by Monjoy Saha                    #
#        monjoybme@gmail.com                    #
#          02 July 2020                         #
#                                               #
#################################################


import numpy as np
import os
import cv2
import glob
from glob import glob
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from skimage import measure, morphology
from skimage.morphology import ball, binary_closing
from skimage.measure import label, regionprops
import copy
from skimage import measure, morphology, segmentation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage as ndimage
import mahotas
from PIL import ImageMath
#from skimage import ndimage
import shutil
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.morphology import watershed as skwater

import pdb

data_path = "/Users/monjoysaha/Downloads/CT_lung_segmentation-master/CT-02_pngs/"
internal_path = "/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-02_Results/step_1_original_to_binary/"
#external_path = "/Users/monjoysaha/Downloads/CT_lung_segmentation-master/study_0009/results/external_lung/"
g = glob(data_path + "/*.png")



def generate_markers(image):
    #Creation of the internal Marker
    marker_internal = image < 20
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    #Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    #Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((512, 512), dtype=np.int)
    #pdb.set_trace()
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    
    return marker_internal, marker_external, marker_watershed

##############################################################
#get list of pngs source files
source_files = os.listdir(data_path)
for file_single in source_files:
  g = glob(data_path + file_single + "/*.png")
  for image in g:
    img = cv2.imread(image)
    #print(image)
    fname = os.path.basename(image)
    #Convert into the gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)
    #################

    # Thresholding
    ret,thresh = cv2.threshold(gray,117,255,cv2.THRESH_BINARY)
    #Show some example markers from the middle
    test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(thresh)

    # Fill holes within the binary image
    img_fill_holes = ndimage.binary_fill_holes(test_patient_internal).astype(int)

    if not os.path.exists(internal_path + file_single):
      os.makedirs(internal_path + file_single)
      print("Created ouput directory: " + internal_path + file_single)

    result = Image.fromarray((img_fill_holes * 255).astype(np.uint8))
    result.save(internal_path + file_single +'/'+fname)






    
    #ret,thresh = cv2.threshold(gray,150,157,cv2.THRESH_BINARY)
    #thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,5)
    #cv2.imshow('image', thresh)
    
    #### fills hole
    #des = cv2.bitwise_not(np.float32(test_patient_internal))
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
    #new = cv2.morphologyEx(des,cv2.MORPH_OPEN,kernel)
    #pdb.set_trace()
    
    #plt.imshow(new, cmap='gray')
    #plt.show()

    #########



    #
    # Noise removal using Morphological
    # closing operation
    #kernel = np.ones((3, 3), np.uint8)
    #closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)
    # Background area using Dialation
    #bg = cv2.dilate(closing, kernel, iterations = 1)
    # Finding foreground area
    #dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
    #ret, fg = cv2.threshold(dist_transform, 0.02 * dist_transform.max(), 255, 0)
    #cv2.imshow('image', fg) 
    
    #im = Image.fromarray(fg)
    #im = im.convert("L")
    #im.save(output_path+fname)

    ######
    # An "interface" to matplotlib.axes.Axes.hist() method
    #n, bins, patches = plt.hist(x=fg)
    #plt.grid(axis='y', alpha=0.75)
    #plt.xlabel('Value')
    #plt.ylabel('Frequency')
    #plt.title('My Very Own Histogram')
    #plt.text(23, 45, r'$\mu=15, b=3$')
    #plt.show()
    #maxfreq = n.max()
    # Set a clean upper y-axis limit.
    #plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    #pdb.set_trace()

    ##############
    #colormask = np.zeros(img.shape, dtype=np.uint8)
    #colormask[thresh!=0] = np.array((0,0,255))
    #blended = cv2.addWeighted(img,0.7,colormask,0.1,0)

    ############
    #Show some example markers from the middle
    #test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(thresh)  # we can aslo use "fg"
    #print ("Internal Marker")
    #plt.imshow(test_patient_internal, cmap='gray')
    #plt.show()
    #print ("External Marker")
    #plt.imshow(test_patient_external, cmap='gray')
    #plt.show()
    #print ("Watershed Marker")
    #plt.imshow(test_patient_watershed, cmap='gray')
    #plt.show()
    #pdb.set_trace()
    #test_patient_internal = cv2.floodFill(test_patient_internal, mask, (0,0), 255);

    # set destination folder
    #if not os.path.exists(internal_path + file_single):    
        #os.makedirs(internal_path + file_single)
        #print("Created ouput directory: " + internal_path + file_single)
    
    #if not os.path.exists(internal_path + file_single):
      #os.makedirs(internal_path + file_single)
      #print("Created ouput directory: " + internal_path + file_single)
    


    
    #result = Image.fromarray((test_patient_internal * 255).astype(np.uint8))
    #pdb.set_trace()
    #result.save(internal_path + file_single +'/'+fname)

    #result_external = Image.fromarray((test_patient_external * 255).astype(np.uint8))
    #result_external.save(external_path+fname)

    


    #colormask = np.zeros(img.shape, dtype=np.uint8)
    #colormask[test_patient_internal!=0] = np.array((0,0,255))
    #blended = cv2.addWeighted(img,0.7,colormask,0.1,0)

    #result_color = Image.fromarray((blended * 255).astype(np.uint8))
    #result_color.save(internal_path+fname)


    




















    

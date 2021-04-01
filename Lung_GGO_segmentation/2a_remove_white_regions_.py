"""Remove white non-GGO regions from original segmented mask"""
#################################################
# 1b_remove_white_regions.py for Python 3          #
#             #
#                                               #                       
#     Written by Monjoy Saha                    #
#        monjoybme@gmail.com                    #
#          02 Sept. 2020                         #
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
original_image_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/CT-02_pngs/'
original_mask_path = "/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-02_Results/original_lung_mask/"
save_path = "/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-02_Results/2a_remove_white_regions/"
#external_path = "/Users/monjoysaha/Downloads/CT_lung_segmentation-master/study_0009/results/external_lung/"
#g = glob(data_path + "/*.png")


##############################################################
#get list of pngs source files
source_files_original = os.listdir(original_image_path)
source_files_mask = os.listdir(original_mask_path)

for single_file in source_files_original: 
  for file_single in source_files_mask:
    if single_file == file_single:
      images = glob(original_image_path + single_file + "/*.png")
      masks= glob(original_mask_path + file_single + "/*.png")
      for image in images:
        fname_image_original = os.path.basename(image)
        for msk in masks:
          fname_image_mask = os.path.basename(msk)
          if fname_image_original == fname_image_mask:
            img_read = cv2.imread(image)
            img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)   # Gray conversion
            msk_read = cv2.imread(msk)
            gray = cv2.cvtColor(msk_read,cv2.COLOR_BGR2GRAY)

            # Thresholding
            ret0,thresh0 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
            # Thresholding1
            ret1,thresh1 = cv2.threshold(gray,130,255,cv2.THRESH_BINARY)

            thresh = thresh0 - thresh1
            Org_mask = cv2.bitwise_and(img_read, img_read, mask=thresh)

            if not os.path.exists(save_path + file_single):
                os.makedirs(save_path + file_single)
                print("Created ouput directory: " + save_path + file_single)

            result = Image.fromarray((Org_mask).astype(np.uint8))
            result.save(save_path + file_single +'/'+fname_image_original)



#for file_single in source_files:
  #g = glob(original_mask_path + file_single + "/*.png")
  #for image in g:
    #img = cv2.imread(image)
    ##print(image)
    #fname = os.path.basename(image)
    ##Convert into the gray
    ##gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ##gray = cv2.equalizeHist(gray)
    #################

    ## Thresholding
    #ret0,thresh0 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    ## Thresholding1
    #ret1,thresh1 = cv2.threshold(gray,130,255,cv2.THRESH_BINARY)

    #thresh = thresh0 - thresh1

    


    


    
    

    #if not os.path.exists(internal_path + file_single):
      #os.makedirs(internal_path + file_single)
      #print("Created ouput directory: " + internal_path + file_single)

    #result = Image.fromarray((thresh).astype(np.uint8))
    #result.save(save_path + file_single +'/'+fname)






    
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


    




















    

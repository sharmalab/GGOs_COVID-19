""" This code can be used for GGO segmentation using K-means clustering"""
#################################################
# 4_K-means_segment_GGO.py for Python 3   #
# extract Only original lung from original      #
#image and binary mask                          s#
#                                               #                       
#     Written by Monjoy Saha                    #
#        monjoybme@gmail.com                    #
#          02 July 2020                         #
#                                               #
#################################################

import numpy as np
import cv2
import os
import glob
from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
from skimage import morphology
import matplotlib.image as mpimg
import pdb
def color_quantization(image, k):
    """Performs color quantization using K-means clustering algorithm"""

    # Transform image into 'data':
    data = np.float32(image).reshape((-1, 3))
    # print(data.shape)

    # Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
    # In this case the maximum number of iterations is set to 20 and epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    # Apply K-means clustering algorithm:
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    #print(center)

    # At this point we can make the image with k colors
    # Convert center to uint8:
    center = np.uint8(center)
    # Replace pixel values with their center value:
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result
def remove_small_objects(img, min_size=150):
        # find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        # connectedComponentswithStats yields every seperated component with information on each of them, such as size
        # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # your answer image
        img2 = img
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] < min_size:
                img2[output == i + 1] = 0
        return img2
##############################################################################################      
heatMap_image_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-03_results/CT-03_Heatmap_step-3/'
save_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-03_results/CT-03_only_GGO_step-4/'
#g= glob(heatMap_image_path + "/*.png")
##############################################################################################
#get list of pngs source files
source_files = os.listdir(heatMap_image_path)
for file_single in source_files:
    g = glob(heatMap_image_path + file_single + "/*.png")
    for image in g:
        fname_image = os.path.basename(image)
        img = cv2.imread(image)
        Z = np.float32(img.reshape((-1,3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 4
        _,labels,centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        #print(centers)
        labels = labels.reshape((img.shape[:-1]))
        reduced = np.uint8(centers)[labels]
        #########################
        blue_dis = 99999999
        blue_center = -1
        b = (255,50,0)
        for i, c in enumerate(centers):
            dis = (c[0]-b[0])**2 + (c[1]-b[1])**2 + (c[1]-b[1])**2
            if dis < blue_dis:
                blue_center = i
                blue_dis = dis
        ##########################
        for i, c in enumerate(centers):
            if i!=blue_center:
                continue
            mask = cv2.inRange(labels, i, i)
            mask = np.dstack([mask]*3) # Make it 3 channel
            ex_img = cv2.bitwise_and(img, mask)
            ex_reduced = cv2.bitwise_and(reduced, mask)
            ex_reduced=cv2.cvtColor(ex_reduced,cv2.COLOR_BGR2RGB)
            result = color_quantization(ex_reduced, k=3)
            gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
            ret1,thresh1 = cv2.threshold(gray,150,155,cv2.THRESH_BINARY)
            ret2,thresh2 = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
            result = thresh2-thresh1
            result = result == 255
            result = morphology.remove_small_objects(result, min_size=500, connectivity=64)
            if not os.path.exists(save_path+ file_single):
                os.makedirs(save_path+ file_single)
                print("Created ouput directory: " + save_path+ file_single)

            mpimg.imsave(save_path+file_single +'/'+fname_image, result, cmap='gray')
    

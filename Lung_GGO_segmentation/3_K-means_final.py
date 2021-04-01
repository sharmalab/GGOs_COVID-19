""" Color based K-means"""

import numpy as np
import cv2
import os
import glob
from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
from skimage import morphology
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
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
    result = result.reshape(image.shape)
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


original_image_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/CT-03_pngs/'      
OrgMask_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-03_results/original_lung_mask/'
save_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-03_results/original_results/'


#get list of png source files
source_files_original = os.listdir(original_image_path)
source_files_mask = os.listdir(OrgMask_path)

for single_file in source_files_original: 
  for mask in source_files_mask:
    if single_file == mask:
      images = glob(original_image_path + single_file + "/*.png")
      masks= glob(OrgMask_path + mask + "/*.png")
      for image in images:
        fname_image_original = os.path.basename(image)
        for msk in masks:
          fname_image_mask = os.path.basename(msk)
          if fname_image_original == fname_image_mask:
            img_Org = cv2.imread(image)
            img_mask = cv2.imread(msk)
            ###############################
            equ = cv2.cvtColor(img_mask,cv2.COLOR_BGR2GRAY)
            #equ = cv2.equalizeHist(equ)
            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            #equ = clahe.apply(equ)
            #res = np.hstack((img,equ)) #stacking images side-by-side
            #plt.imshow(equ)
            #plt.show()
            #pdb.set_trace()
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            Z = np.float32(img_mask.reshape((-1,3)))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 3
            _,labels,centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            #print(centers)
            labels = labels.reshape((img_mask.shape[:-1]))
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
                ex_img = cv2.bitwise_and(img_mask, mask)
                ex_reduced = cv2.bitwise_and(reduced, mask)
                plt.imshow(ex_reduced, cmap='gray')
                plt.show()
                ex_reduced=cv2.cvtColor(ex_reduced,cv2.COLOR_BGR2RGB)
                result = color_quantization(ex_reduced, k=3)
                gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
                ret1,thresh1 = cv2.threshold(gray,110,155,cv2.THRESH_BINARY)
                ret2,thresh2 = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
                result = thresh2-thresh1
                result = result == 255
                result = morphology.remove_small_objects(result, min_size=440, connectivity=1)
                result = ndimage.binary_fill_holes(result).astype(np.uint8)
                ##########################
                """ Segment Only GGO regions """
                #################################
                ori_GGO = cv2. bitwise_and(img_mask, img_mask,mask=result)
                #plt.imshow(ori_GGO, cmap='gray')
                #plt.show()
                #plt.hist(ori_GGO.ravel(),256,[0,256]); plt.show()
                ####################################
                contours, hierarchy =  cv2.findContours(result.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_Org, contours, -1, (255,0,0), 1)
                if not os.path.exists(save_path + single_file):
                    os.makedirs(save_path + single_file)
                    print("Created ouput directory: " + save_path + single_file)

                mpimg.imsave(save_path+single_file+'/'+fname_image_original, img_Org, cmap='gray')
                #plt.imshow(img_read, cmap='gray')
                #plt.show()
  
    
    

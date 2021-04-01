#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#SImple image segmentation using K-Means clustering algo

#color clustering
#Image segmentation from video using OpenCV and K-means clustering
import numpy as np
import cv2
import matplotlib.pyplot as plt
import collections
import pdb
from PIL import Image
from skimage import morphology
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
    print(center)

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

##############################
#original_image_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/check/study_0255/'
#Kmeans_image = cv2.imread("/Users/monjoysaha/Downloads/CT_lung_segmentation-master/check/heat_map/"
#save_GGO_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/check/only_GGO'
#g = glob(original_image_path + "/*.png")
#m= glob(mask_image_path + "/*.png")

original_image = cv2.imread("/Users/monjoysaha/Downloads/CT_lung_segmentation-master/check/heat_map_new/study__z016.png")

# Converting from BGR Colours Space to HSV
img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
result = color_quantization(img, k=3)

gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
ret1,thresh1 = cv2.threshold(gray,150,155,cv2.THRESH_BINARY)
ret2,thresh2 = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

result = thresh2-thresh1
result =result ==255
#plt.imshow(result, cmap='gray')
#plt.show()

################


cleaned = morphology.remove_small_objects(result, min_size=400, connectivity=4)
#plt.imshow(cleaned, cmap='gray')
#plt.show()
img3 = np.zeros((cleaned.shape)) # create array of size cleaned
img3[cleaned > 0] = 255 
img3= np.uint8(img3)



cv2.imshow("cleaned", img3)
cv2.imwrite("cleaned.jpg", img3)
cv2.waitKey(0)





#vals = gray.mean(axis=0).flatten()
#counts, bins = np.histogram(vals, range(257))
# plot histogram centered on values 0..255
#plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
#plt.xlim([-0.5, 255.5])
#plt.show()



#pdb.set_trace()
#result = Image.fromarray((result * 255).astype(np.uint8))
#result.save('result.png')
#

#dc = get_dominant_color(img, clusters=3, attempts=10)

#plt.imshow(dc)
#plt.show()



""" This code can be used to generate Heat map"""
#################################################
# heat_map.py for Python 3                      #
# Heat map generation                           #
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
from matplotlib import pyplot as plt
from PIL import Image
import pdb
from skimage import io
from skimage import measure, morphology
from skimage.morphology import ball, binary_closing
from skimage.measure import label, regionprops
import copy
from skimage import measure, morphology, segmentation
from PIL import Image, ImageOps
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
from skimage.transform import resize
import matplotlib.image as mpimg
from skimage import exposure

######################################################
# sample the colormaps that you want to use. Use 128 from each so we get 256
# colors in total
colors1 = plt.cm.binary(np.linspace(0., 1, 128))
colors2 = plt.cm.gist_heat_r(np.linspace(0, 1, 128))

# combine them and build a new colormap
colors = np.vstack((colors1, colors2))
mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

#######################################################


data_path = "/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-03_results/step_2_originalMask/"
output_path = "/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/CT-03_results/CT-03_Heatmap_step-3/"
g = glob(data_path + "/*.png")

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
        #plt.show()
        #pdb.set_trace()
        #################
        #gray=np.transpose(gray)
        #gray=np.rot90(gray,axes=(-2,-1))
        #plt.pcolor(gray, cmap=mymap)
        #plt.colorbar()
        ##########################
        #plt.imshow(gray)
        #plt.set_cmap('seismic')
        ###########################
        #plt.show()
        plt.axis('off')
        plt.margins(0,0)
        #plt.savefig(output_path+fname, transparent = True, bbox_inches = 'tight', pad_inches = 0)
        if not os.path.exists(output_path + file_single):
            os.makedirs(output_path + file_single)
            print("Created ouput directory: " + output_path + file_single)
            #pdb.set_trace()
        mpimg.imsave(output_path+file_single+'/'+fname, gray, cmap='hsv')

   
    

    

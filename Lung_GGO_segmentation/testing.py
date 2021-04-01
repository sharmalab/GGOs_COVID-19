"""https://stackoverflow.com/questions/49834264/mri-brain-tumor-image-processing-and-segmentation-skull-removing"""
""" https://medium.com/@hengloose/a-comprehensive-starter-guide-to-visualizing-and-analyzing-dicom-images-in-python-7a8430fcb7ed """
""" This code can be used to generate binary images from the original PNGs"""

import numpy as np
import os
import cv2
import glob
from glob import glob
#import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import pdb
from skimage import io
# skimage image processing packages
from skimage import measure, morphology
from skimage.morphology import ball, binary_closing
from skimage.measure import label, regionprops
import copy
from skimage import measure, morphology, segmentation
from PIL import Image, ImageOps
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
from skimage.transform import resize
data_path = "/Users/monjoysaha/Downloads/CT_lung_segmentation-master/check/original/test/"
output_path = "/Users/monjoysaha/Downloads/CT_lung_segmentation-master/check/heat_map_new/"
g = glob(data_path + "/*.png")

# sample the colormaps that you want to use. Use 128 from each so we get 256
# colors in total
colors1 = plt.cm.binary(np.linspace(0., 1, 256))
colors2 = plt.cm.gist_heat_r(np.linspace(0, 1, 256))

# combine them and build a new colormap
colors = np.vstack((colors1, colors2))
mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)


for image in g:
    img = cv2.imread(image)
    #print(image)
    fname = os.path.basename(image)
    #Convert into the gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#################
    gray=np.transpose(gray)
    gray=np.rot90(gray,axes=(-2,-1))
    plt.pcolor(gray, cmap=mymap)
    #plt.colorbar()
    cv2.imwrite('13.png', gray)
    #pdb.set_trace()
##########################
    #plt.imshow(gray)
    #plt.set_cmap('seismic')
 ###########################   
    #plt.show()
    plt.axis('off')
    plt.savefig(output_path+fname, bbox_inches='tight', pad_inches=0, orientation=u'vertical', dpi=100)

    #

    

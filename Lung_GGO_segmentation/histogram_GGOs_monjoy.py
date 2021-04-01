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


OrgMask_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Histogram_GGO/Segmented_GGO_regions/'
save_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Histogram_GGO/histogram_results/'


#get list of png source files

source_files_mask = os.listdir(OrgMask_path)


for mask in source_files_mask:
  masks= glob(OrgMask_path + mask + "/*.png")
  for msk in masks:
      fname_image_mask = os.path.basename(msk)
      img_mask = cv2.imread(msk)
      plt.hist(img_mask.ravel(),256,[0,256])
      plt.ylim((None, 500))
      if not os.path.exists(save_path + mask):
          os.makedirs(save_path + mask)
          print("Created ouput directory: " + save_path + mask)
      plt.savefig(save_path+mask+'/'+fname_image_mask)
      plt.clf()
      plt.close()
        


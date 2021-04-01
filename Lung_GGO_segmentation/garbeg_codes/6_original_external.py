import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import os

import glob
from glob import glob
import pandas as pd

from PIL import Image
import pdb
#skimage image processing packages
from skimage import measure, morphology
from skimage.morphology import ball, binary_closing
from skimage.measure import label, regionprops
import copy
from skimage import measure, morphology, segmentation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage as ndimage
import mahotas

def ShowImage(title,img,ctype):
  plt.figure(figsize=(10, 10))
  if ctype=='bgr':
    b,g,r = cv2.split(img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    plt.imshow(rgb_img)
  elif ctype=='hsv':
    rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    plt.imshow(rgb)
  elif ctype=='gray':
    plt.imshow(img,cmap='gray')
  elif ctype=='rgb':
    plt.imshow(img)
  else:
    raise Exception("Unknown colour type")
  plt.axis('off')
  plt.title(title)
  plt.show()

##############################
original_image_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/CT-01_pngs/study_0258/'
mask_image_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Analysis-29July/save_results/'
save_path = '/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Analysis-29July/save_results/'
g = glob(original_image_path + "/*.png")
m= glob(mask_image_path + "/*.png")

for image in g:
  fname_image = os.path.basename(image)
  for mask in m:
    fname_mask = os.path.basename(mask)
    if fname_image == fname_mask:
      img = cv2.imread(image)
      msk = cv2.imread(mask)
      result = cv2.subtract(msk, img)
      #plt.imshow(img)
      #plt.show()
      #pdb.set_trace()
      # Mask input image with binary mask
      #result = cv2.bitwise_and(img, msk)
      # Color background white
      #result[msk==0] = 255 # Optional
      result = Image.fromarray((result * 255).astype(np.uint8))
      result.save(save_path+fname_image)
      
      
    
    
  
 





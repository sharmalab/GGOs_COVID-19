"""https://stackoverflow.com/questions/49834264/mri-brain-tumor-image-processing-and-segmentation-skull-removing"""
""" This code can be used to segment main lung from binary images"""
import numpy as np
import os
import cv2
import glob
from glob import glob
import pandas as pd
from matplotlib import pyplot as plt
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
from PIL import ImageMath

data_path = "/Users/monjoysaha/Downloads/CT_lung_segmentation-master/CT-01_pngs/study_0256/"
internal_path = "/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/study_0256/external/"
#external_path = "/Users/monjoysaha/Downloads/CT_lung_segmentation-master/Final_GGO_Segmentation_results/study_0256/external/"
g = glob(data_path + "/*.png")


import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.morphology import watershed as skwater

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

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    #binary_image = np.array(image >= -200, dtype=np.int8)+1
   
    binary_image  = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    labels = measure.label(binary_image)
    #pdb.set_trace()
    background_label = labels[0,0,0]
 
    # Fill the air around the person
    binary_image[background_label == labels] = 2
    if fill_lung_structures:
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1 
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            if l_max is not None: 
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
 
    # Remove other air pockets inside body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image
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

for image in g:
    img = cv2.imread(image)
    #print(image)
    fname = os.path.basename(image)
    #Convert into the gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    #################

    # Thresholding
    #ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret,thresh = cv2.threshold(gray,150,157,cv2.THRESH_BINARY)
    #cv2.imshow('image', thresh)

    #pdb.set_trace()
    # Noise removal using Morphological
    # closing operation
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)
    # Background area using Dialation
    bg = cv2.dilate(closing, kernel, iterations = 1)
    # Finding foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
    ret, fg = cv2.threshold(dist_transform, 0.02 * dist_transform.max(), 255, 0)
    #cv2.imshow('image', fg) 
    
    im = Image.fromarray(fg)
    im = im.convert("L")
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
    test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(thresh)  # we can aslo use "fg"
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

    
    result = Image.fromarray((test_patient_internal * 255).astype(np.uint8))
    result.save(internal_path+fname)

    #result_external = Image.fromarray((test_patient_external * 255).astype(np.uint8))
    #result_external.save(external_path+fname)



    colormask = np.zeros(img.shape, dtype=np.uint8)
    colormask[test_patient_internal!=0] = np.array((0,0,255))
    blended = cv2.addWeighted(img,0.7,colormask,0.1,0)

    #result_color = Image.fromarray((blended * 255).astype(np.uint8))
    #result_color.save(internal_path+fname)




















    

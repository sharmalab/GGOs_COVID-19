# AI-DRIVEN QUANTIFICATION OF GROUND GLASS OPACITIES IN 3D COMPUTED TOMOGRAPHY IMAGES OF LUNGS OF COVID-19 PATIENTS

[![GitHub stars](https://img.shields.io/github/stars/sharmalab/GGOs_COVID-19)](https://github.com/sharmalab/GGOs_COVID-19/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/sharmalab/GGOs_COVID-19)](https://github.com/sharmalab/GGOs_COVID-19/issues)
[![GitHub forks](https://img.shields.io/github/forks/sharmalab/GGOs_COVID-19)](https://github.com/sharmalab/GGOs_COVID-19/network)
![GitHub License](https://img.shields.io/github/license/sharmalab/GGOs_COVID-19)

# Data
We used "**MosMedData: Chest CT Scans with COVID-19 Related Findings**" (https://mosmed.ai/datasets/covid19_1110). Data needs to download before using this pipeline. The size of the data is 33 GB. All the data is in ```.nii``` format. Hence, you need to convert them into ```.png``` for using this pipeline. You can use ```nii_to_pngs_converter.py``` for this purpose. 


# Lung and GGOs Segmentation
This is the first step of this pipeline. All the relevant codes have been stored inside the folder "**Lung_GGO_segmentation**"

## ```1_original_to_binary.py``` first script of this pipeline. 
This code has been used for generating binary masks from original images. ```data_path``` is the original data path.  The output of this code will be saved at ```internal_path```.  

## ```1a_convex_hull_CT_monjoy.py``` second script of this pipeline
This code has been used for getting convex hull, convex points. The input is binary images. ```data_path``` is a path of dataset. The results will be saved at ```save_path```. 

## ```2_Original_lung_from_binary.py``` third script of this pipeline
This code can be used to extract the only lung from the original images and binary mask (resulted from step-1 code). ```original_image_path``` is an original image path. ```mask_image_path``` is a path of masks. All the results will be saved at ```save_path```. 

## 2a_remove_white_regions_.py fourth script of this pipeline
This code removes white non-GGO regions from the original segmented images. ```original_image_path``` is an original image path. ```original_mask_path``` is a path of masks. All the results will be saved at ```save_path```. 

## ```3_K-means_final.py``` fifth script of this pipeline
```3_K-means_final.py``` is used to segment GGOs. ```original_image_path ``` is the original image path. ```OrgMask_path``` is the mask path. All the results will be saved at ```save_path```. 

# Point Cloud
## ```2a_convert_monjoy.py``` first script of this pipeline
To convert stack of images into ```point cloud``` use ```2a_convert_monjoy.py```. The code is stored inside the folder ```1_Convert_images_to_point_cloud```. In our study, we used sixteen consecutive images. You can use as many images as you want. There are no restrictions. Set image path ```image_path``` and save results at ```save_path```. All the final files will be saved in ```.ply``` format. 

## ```visualize_pcd_file.py``` second script of this pipeline
This script can be used to visualize point clouds. 

## ```downsample_main_code_24dec.py``` third script of this pipeline
To downsample the number of points of the point cloud, you can use ```downsample_main_code_24dec.py```. The code is stored inside the folder ```2_Downsample_pointCloud```.  ```ply_file_path``` denotes ```.ply``` file path and ```downsampled_files_save_path``` represents the path, where downsampled files will be saved. In our study, we downsampled the number of points from 800K to 2480 points. 

## ```3_Ply_to_pcd_monjoy``` fourth script of this pipeline
This script is used for converting from ```.ply``` format to ```.pcd``` format. ```.ply``` and ```.pcd``` both files represent point clouds. For our easy analysis, we converted all ```.ply``` to ```.pcd```. The script is stored inside the folder ```3_Ply_to_pcd_monjoy```. 

## Convert .pcd to ```.txt``` format fifth script of this pipeline
For the ```PointNet++``` data needs to be converted ```.txt``` format. Hence, we used a script inside ```4_prepare_data_for_pointNet/pointcloudToTXT-master/build/``` folder. 
Use the below command for this purpose.
``` cd /build ```
For ```.pcd``` files
```./pointcloudToTXT <pcd file> -o <output dir>```
For ```.ply``` files
```./pointcloudToTXT <ply file> -o <output dir>```





# PointNet++


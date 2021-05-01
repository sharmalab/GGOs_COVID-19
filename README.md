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

# Point Cloud

# PointNet++


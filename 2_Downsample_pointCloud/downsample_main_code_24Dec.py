#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 19:10:23 2020

@author: ms
"""

import open3d
import os
import numpy as np

import pdb
#dense_pcd_path = "/home/ms/Desktop/Point_Cloud_Monjoy/3_Downsample_pointCloud/dataset/semantic_raw/new_modified.pcd"
#sparse_pcd_path = "/home/ms/Desktop/Point_Cloud_Monjoy/3_Downsample_pointCloud/dataset/semantic_downsampled/downsampled_new_modified.pcd"

ply_file_path = "/home/ms/Desktop/Point_Cloud_Monjoy/data_19Jan2021/2_ply_data/"
downsampled_files_save_path = "/home/ms/Desktop/Point_Cloud_Monjoy/data_19Jan2021/3_downsampled_ply/"

source_files = os.listdir(ply_file_path)


def down_sample(
    dense_pcd_path, voxel_size
):
  
    # Inputs
    dense_pcd = open3d.io.read_point_cloud(dense_pcd_path)
    

    # Skip label 0, we use explicit frees to reduce memory usage
    print("Num points:", np.asarray(dense_pcd.points).shape[0])
    
    # Downsample points
    min_bound = dense_pcd.get_min_bound() - voxel_size * 0.5
    max_bound = dense_pcd.get_max_bound() + voxel_size * 0.5
    sparse_pcd = open3d.geometry.PointCloud.voxel_down_sample(
        dense_pcd, voxel_size
    )
    print("Num points after down sampling:", np.asarray(sparse_pcd.points).shape[0])
    
    open3d.io.write_point_cloud(downsampled_files_save_path+single_file, sparse_pcd)
    #open3d.io.write_point_cloud("lung_down.pcd", sparse_pcd)
    
    #print("Point cloud written to:", sparse_pcd_path)

if not os.path.exists(downsampled_files_save_path):
    os.makedirs(downsampled_files_save_path)
    print("Created ouput directory: " + downsampled_files_save_path)

if __name__ == "__main__":
    #voxel_size = 0.05
    voxel_size = 0.5
    for single_file in  source_files:
        if single_file.endswith(".ply"):
           #print(single_file)
           down_sample(ply_file_path + single_file, voxel_size)
        
            
           #pdb.set_trace()
    # By default
    # raw data: "dataset/semantic_raw"
    # downsampled data: "dataset/semantic_downsampled"
    #current_dir = os.path.dirname(os.path.realpath(__file__))
    #dataset_dir = os.path.join(current_dir, "dataset")
    #raw_dir = os.path.join(dataset_dir, "semantic_raw")
    #downsampled_dir = os.path.join(dataset_dir, "semantic_downsampled")

    # Create downsampled_dir
    #os.makedirs(downsampled_dir, exist_ok=True)
    #down_sample(dense_pcd_path, voxel_size)
    


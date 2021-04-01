import open3d as o3d
import os
import pdb
ply_file_path ="/home/ms/Desktop/Point_Cloud_Monjoy/data_19Jan2021/3_downsampled_ply/"
save_pcd_file_path = "/home/ms/Desktop/Point_Cloud_Monjoy/data_19Jan2021/4_downsampled_pcd/"

source_files= os.listdir(ply_file_path)
if not os.path.exists(save_pcd_file_path):
    os.makedirs(save_pcd_file_path)
    print("Created ouput directory: " + save_pcd_file_path)

for single_file in source_files:
    if single_file.endswith(".ply"):
       fname = os.path.basename(single_file)
       base, ext = os.path.splitext(fname)
       print(fname)
       #pdb.set_trace()
       pcd = o3d.io.read_point_cloud(ply_file_path+ single_file)
       o3d.io.write_point_cloud(save_pcd_file_path+base+".pcd",pcd, write_ascii=False, compressed=False, print_progress=False)

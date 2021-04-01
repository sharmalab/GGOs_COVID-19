import numpy as np
import open3d as o3d

# Read .ply file
#input_file = "pumpkin.ply"
#pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud

# Visualize the point cloud within open3d
#o3d.draw_geometries([pcd]) 

# Convert open3d format to numpy array
# Here, you have the point cloud in numpy format. 
#point_cloud_in_numpy = np.asarray(pcd.points)


print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("/home/ms/Desktop/Point_Cloud_Monjoy/3_Downsample_pointCloud/dataset/semantic_downsampled/9059_points/new_modified.pcd")
#print(pcd)
#print(np.asarray(pcd.points))
downpcd = pcd.voxel_down_sample(voxel_size=0.8)
print(np.asarray(downpcd.points))
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

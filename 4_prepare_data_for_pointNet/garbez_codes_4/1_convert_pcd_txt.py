import numpy as np
import open3d as o3d
import pdb
pcd= o3d.io.read_point_cloud("new_modified.ply")
array=np.asarray(pcd.points)

with open("points_downsample.txt", mode='w') as f:  # I add the mode='w'
    for i in range(len(array)):
        pdb.set_trace()
        f.write("%f    "%float(array[i][0].item()))
        f.write("%f    "%float(array[i][1].item()))
        f.write("%f    "%float(array[i][2].item()))
        f.write("%f    \n"%float(array[i][3].item()))
        #f.write("%f    "%float(array[i][4].item()))
        #f.write("%f    \n"%float(array[i][5].item()))

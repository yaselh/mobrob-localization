import os
import pcl
import numpy as np
import glob

relative_pointcloud_data_path = '../../../../data/pointclouds5'
pointcloud_data_path = os.path.join(os.path.dirname(__file__), relative_pointcloud_data_path)

if __name__=='__main__':
    print('Running pointcloud data converter...')
    files = glob.glob(pointcloud_data_path + '/*.pcd')
    for file in files:
        print(file)
        pointcloud = pcl.load_XYZRGB(file)
        pointcloud_np = pointcloud.to_array()
        print(pointcloud_np.dtype)
        np.save(os.path.splitext(file)[0] + '.npy', pointcloud_np, allow_pickle=False)

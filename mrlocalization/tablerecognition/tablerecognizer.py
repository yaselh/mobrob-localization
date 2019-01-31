# See http://pointclouds.org/documentation/tutorials/planar_segmentation.php
import os

import pcl
import pcl.pcl_visualization

import numpy as np

import math

import matplotlib.pyplot as plt

import sys
relative_bbox_calc_dir = '../utils'
bbox_calc_dir = os.path.join(os.path.dirname(__file__), relative_bbox_calc_dir)
sys.path.append(bbox_calc_dir)

from bbox_calculator import minimum_bounding_rectangle

def euclidean_distance(list1, list2):
    dist = math.sqrt((list2[0]-list1[0])**2 + (list2[1]-list1[1])**2 + (list2[2]-list1[2])**2)
    return dist


class TableRecognizer:

    # Code taken from https://stackoverflow.com/questions/41606471/3d-coordinates-to-2d-plane-python
    def rotate_to_xy(self, coords, plane):
        """
        Rotates the points from reference plane to xy axis
        Returns the new coordinates after rotation
        """
        plane_axis = np.array(plane)/ np.linalg.norm(plane) # Normalize plane axis
        transform_axis = np.array([0,0,1])                  # X-Y Plane

        # Axis along which we should rotate to reach XY Plane
        rotation_axis = np.cross(plane_axis, transform_axis)

        # Rotation angle to reach XY plane - Essentially angle of plane to XY Axis
        angle_rotation = np.arccos(np.dot(plane_axis, transform_axis) /
                                   (np.linalg.norm(plane_axis) * np.linalg.norm(transform_axis)))

        print("Rotation axis for {} is {} and angle is {} ".format(plane, rotation_axis, angle_rotation))

        # Compute rotation matrix to perform translation
        rotation_matrix = self.__rotation_matrix(rotation_axis, angle_rotation)

        # Compoute coordinates after rotation
        projected = np.dot(coords, rotation_matrix.T)
        return projected, rotation_axis, angle_rotation

    def __rotation_matrix(self, axis, theta):
        """
        Compute Rotation Matrix associated with rotation about give axis by theta radian
        http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
        """
        axis = axis/math.sqrt(np.dot(axis, axis))
        a = math.cos(theta/2.0)
        b, c, d = -axis*math.sin(theta/2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

    # Initialization routine
    def initialize_recognizer(self):
        pass

    def predict_center_point(self, pointcloud):
        # Filter all points that are more than x meters away if not already done
        fil = pointcloud.make_passthrough_filter()
        fil.set_filter_field_name("z")
        fil.set_filter_limits(0, 2.0)
        pointcloud = fil.filter()


        seg = pointcloud.make_segmenter_normals(ksearch=50)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
        seg.set_normal_distance_weight(0.1)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_max_iterations(100)
        seg.set_distance_threshold(0.05)
        # Indices holds all indices of the input pointcloud that lie on the plane.
        # Model holds the model coefficients ax + by + cz + d = 0 so (a, b, c) is the normal vector
        indices, model = seg.segment()

        normal_vector = model[0:3]
        normal_vector_normalized = normal_vector/np.linalg.norm(normal_vector)
        print(normal_vector_normalized)

        cloud_plane = pointcloud.extract(indices, negative=False)

        # Calculate the table bounding box
        plane_pointcloud_np = cloud_plane.to_array()
        # Transform table plane such that is is parallel to the XY plane
        transformed_points, rotation_axis, rotation_angle = self.rotate_to_xy(plane_pointcloud_np[:,:3], model[:3])
        # Calculate minimum rectangle around projection on XY plane
        table_bbox = minimum_bounding_rectangle(transformed_points[:,:2])
        # Calculate mean z value
        mean_z = np.mean(transformed_points, axis=0)[2]
        # Define table corner points
        right_corner = np.append(table_bbox[0], mean_z)
        bottom_corner = np.append(table_bbox[1], mean_z)
        left_corner = np.append(table_bbox[2], mean_z)
        top_corner = np.append(table_bbox[3], mean_z)
        # Transform corner points back to original coordinate system
        corner_points = np.array([right_corner, bottom_corner, left_corner, top_corner])
        back_transformation_rotation = self.__rotation_matrix(rotation_axis, -rotation_angle)
        corner_points_camera_coordinates = np.dot(corner_points, back_transformation_rotation.T)
        right_corner = corner_points_camera_coordinates[0]
        far_corner = corner_points_camera_coordinates[1]
        left_corner = corner_points_camera_coordinates[2]
        close_corner = corner_points_camera_coordinates[3]

        side_1 = np.linalg.norm(right_corner - far_corner)
        side_2 = np.linalg.norm(far_corner - left_corner)
        side_3 = np.linalg.norm(left_corner - close_corner)
        side_4 = np.linalg.norm(close_corner - right_corner)

        center_point = close_corner + 1/2 * (far_corner - close_corner)
        normal_vector = np.array([model[0], model[1], model[2]]) # Not validated... think for test.
        z_axis = -normal_vector / np.linalg.norm(normal_vector) # Note that normal vector is pointing in wrong direction in current configuration so has to be set to minus here
        x_axis = right_corner - center_point
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)


        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for i in range(0, plane_pointcloud_np.shape[0], 500):
        #     point = plane_pointcloud_np[i]
        #     ax.scatter(point[0], point[1], point[2], c='b', marker='o')
        # ax.scatter(center_point[0], center_point[1], center_point[2], c='r', marker='o')
        # ax.quiver(center_point[0], center_point[1], center_point[2], x_axis[0], x_axis[1], x_axis[2])
        # ax.quiver(center_point[0], center_point[1], center_point[2], y_axis[0], y_axis[1], y_axis[2])
        # ax.quiver(center_point[0], center_point[1], center_point[2], z_axis[0], z_axis[1], z_axis[2])
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()

        #point_list = cloud_plane.to_list()

        #eps = 0.1
        #for point in point_list:
        #    if euclidean_distance(point, center_point) < eps:
        #        point[3] = 2.341805152028776e-38



        #p = pcl.PointCloud_PointXYZRGB(point_list)
        #visual = pcl.pcl_visualization.CloudViewing()
        #visual.ShowColorCloud(p, b'cloud')

        #flag = True
        #while flag:
        #    flag != visual.WasStopped()

        return center_point, x_axis, y_axis, z_axis, indices

if __name__ == '__main__':
    print('Running table recognizer...')
    relative_pointcloud_path = '../../data/pointclouds4/pos2_1515576584050514.pcd'
    pointcloud_path = os.path.join(os.path.dirname(__file__), relative_pointcloud_path)
    pointcloud = pcl.load_XYZRGB(pointcloud_path)
    table_recognizer = TableRecognizer()
    table_recognizer.initialize_recognizer()
    table_recognizer.predict_center_point(pointcloud)

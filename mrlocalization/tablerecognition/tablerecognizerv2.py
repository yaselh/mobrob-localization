# See http://pointclouds.org/documentation/tutorials/planar_segmentation.php
import os
import sys

import math
from math import atan2, sin, cos

import numpy as np

import scipy
from scipy import linalg, matrix

from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny

from sklearn.linear_model import RANSACRegressor

import matplotlib.pyplot as plt

from planeregressor import PlaneRegressor
from lineregressor import LineRegressor

class TableRecognizer:

    # Initialization routine
    def initialize_recognizer(self):
        self.left_corner_approx_position = np.array([170, 30])
        self.right_corner_approx_position = np.array([140, 625])


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

        #print("Rotation axis for {} is {} and angle is {} ".format(plane, rotation_axis, angle_rotation))

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

    def get_angle_difference(self, angle1, angle2):
        return atan2(sin(angle1-angle2), cos(angle1-angle2))

    def line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return None

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    def select_valid_lines(self, lines, approximate_intersection_pos):
        approx_pos_np = np.array(approximate_intersection_pos)
        valid_lines = []
        for line0 in lines:
            for line1 in lines:
                intersection = self.line_intersection(line0, line1)
                if intersection is not None:
                    intersection_np = np.array(intersection)
                    distance = np.linalg.norm(approx_pos_np - intersection_np)
                    if distance < 10.0:
                        valid_lines.append((line0, line1, intersection))
        return valid_lines

    # Currently not used cause line is instead calculated in 2D (see beyond)
    # def get_line_from_points(self, line_np, pointcloud_np):
    #     steps = np.arange(0.0, 1.0, 0.005)
    #     start_point = line_np[0]
    #     stop_point = line_np[1]
    #     valid_points = []
    #     for step in steps:
    #         position = start_point + step * (stop_point - start_point)
    #         pointcloud_point = pointcloud_np[int(position[1])][int(position[0])]
    #         # Select points that do not contain nan values and are closer to the camera than 2 meters
    #         if not np.isnan(pointcloud_point).any() and pointcloud_point[2] < 1.5:
    #             valid_points.append(pointcloud_point)
    #     valid_points_np = np.array(valid_points)
    #
    #
    #
    #     lineregressor = LineRegressor()
    #     model = lineregressor.fit(valid_points_np)
    #
    #     point0 = model[0]
    #     vector = model[2]
    #     line = (point0, vector)
    #     return line


    # Function for calculating a line through points in a pointcloud
    def get_line_from_points2d(self, line_np, pointcloud_np, normal_vector):
        steps = np.arange(0.0, 1.0, 0.005)
        start_point = line_np[0]
        stop_point = line_np[1]
        valid_points = np.empty((0,3))
        for step in steps:
            position = start_point + step * (stop_point - start_point)
            pointcloud_point = pointcloud_np[int(position[1])][int(position[0])]
            # Select points that do not contain nan values and are closer to the camera than 2 meters
            if not np.isnan(pointcloud_point).any() and pointcloud_point[2] < 1.5:
                valid_points = np.append(valid_points, [pointcloud_point], axis = 0)
        valid_points_np = np.array(valid_points)
        if valid_points_np.shape[0] <= 2:
            return None

        transformed_points, rotation_axis, rotation_angle = self.rotate_to_xy(valid_points, normal_vector)
        transformed_points_X = transformed_points[:,:1]
        transformed_points_y = transformed_points[:,1:2]

        ransac = RANSACRegressor()
        ransac.fit(transformed_points_X, transformed_points_y)
        X0 = transformed_points_X[0:1,:]
        Xn = transformed_points_X[transformed_points_X.shape[0] - 1:transformed_points_X.shape[0],:]
        y0 = ransac.predict(X0)
        yn = ransac.predict(Xn)


        #print(y0)

        # Create line definition and rotate back
        z_average = np.average(transformed_points, axis = 0)[2]
        point0 = np.array([X0[0][0], y0[0][0], z_average])
        point1 = np.array([Xn[0][0], yn[0][0], z_average])
        back_transformation_rotation = self.__rotation_matrix(rotation_axis, -rotation_angle)
        point0 = np.dot(point0, back_transformation_rotation.T)
        point1 = np.dot(point1, back_transformation_rotation.T)
        vector = point1 - point0
        vector /= np.linalg.norm(vector)

        #for point in transformed_points:
        #    plt.scatter(point[0], point[1], c='b')
        #plt.plot([X0[0],Xn[0]],[y0[0],yn[0]])
        #plt.show()
        #print(point0)
        line = (point0, vector)
        return line




    def predict_center_point(self, pointcloud_np, rgb_image):


        # -------------- Find normal vector of the table plate --------------
        # Filter points that are nan or too far away
        filtered_pointcloud = pointcloud_np[np.logical_not(np.isnan(pointcloud_np[:,2]))]
        filtered_pointcloud = filtered_pointcloud[filtered_pointcloud[:,2] < 2.0]
        # Perform Ransac Regression
        planeregressor = PlaneRegressor()
        model, inliers = planeregressor.fit(filtered_pointcloud)
        if model is None:
            raise RuntimeError('Table plane could not be recognized.')

        normal_vector = model[:3]

        # If vector points towards floor switch sign
        if normal_vector[2] > 0:
            normal_vector = -normal_vector


        # -------------- Detect table edges via hough transformation --------------
        grayscale_image = rgb2gray(rgb_image)
        edges = canny(grayscale_image, 3.0, 0.15, 0.2)
        lines = probabilistic_hough_line(edges, threshold=100, line_length=125, line_gap=10)


        # -------------- Determine edges that are adjacent to the right table edge --------------
        approximate_line_intersection = (625, 235)
        line_pairs = self.select_valid_lines(lines, approximate_line_intersection)

        if len(line_pairs) == 0:
            raise RuntimeError('No matching line pairs that intersect approximately at {} found.'.format(approximate_line_intersection))


        # -------------- Select line pair whose inner angle is closest to 90 degrees --------------
        min_angle_difference = 2 * math.pi
        best_line0_pcl = None
        best_line1_pcl = None
        for line_information in line_pairs:
            line0 = line_information[0]
            line0_np = np.array(line0)
            line1 = line_information[1]
            line1_np = np.array(line1)

            # Get pointcloud values
            pointcloud_np = np.reshape(pointcloud_np, (480, 640, 4))
            line0_pcl = self.get_line_from_points2d(line0_np, pointcloud_np[:,:,:3], normal_vector)
            line1_pcl = self.get_line_from_points2d(line1_np, pointcloud_np[:,:,:3], normal_vector)

            # Skip if line could not be created from pointcloud
            if line0_pcl is None or line1_pcl is None:
                continue
            v1 = line0_pcl[1]
            v2 = line1_pcl[1]
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            angle_difference = np.abs(self.get_angle_difference(math.pi/2, angle))
            if angle_difference < min_angle_difference:
                min_angle_difference = angle
                best_line0_pcl= line0_pcl
                best_line1_pcl= line1_pcl
        # Check if lines are set and otherwise throw exception
        if best_line0_pcl is None or best_line1_pcl is None:
            raise RuntimeError('No lines could be created from the pointcloud.')



        # -------------- Determine intersection of the lines in 2D space --------------
        p0_l0 = best_line0_pcl[0]
        v_l0 = best_line0_pcl[1]
        p1_l0 = p0_l0 + v_l0

        p0_l1 = best_line1_pcl[0]
        v_l1 = best_line1_pcl[1]
        p1_l1= p0_l1 + v_l1

        transformed_points, rotation_axis, rotation_angle = self.rotate_to_xy(np.array([p0_l0, p1_l0, p0_l1, p1_l1]), normal_vector)
        P_2D = transformed_points[0]
        Q_2D = transformed_points[1]
        R_2D = transformed_points[2]
        S_2D = transformed_points[3]
        intersection_point = self.line_intersection((P_2D, Q_2D), (R_2D,S_2D))
        #plt.plot([P_2D[0], Q_2D[0]], [P_2D[1], Q_2D[1]])
        #plt.plot([R_2D[0], S_2D[0]], [R_2D[1], S_2D[1]])
        #plt.scatter(intersection_point[0], intersection_point[1])
        #plt.show()
        intersection_point_3D = np.array([intersection_point[0], intersection_point[1], P_2D[2]])
        intersection_point_3D = np.reshape(intersection_point_3D, (3,))
        back_transformation_rotation = self.__rotation_matrix(rotation_axis, -rotation_angle)
        intersection_point_3D = np.dot(intersection_point_3D, back_transformation_rotation.T)


        # Gather information for calculation of coordinate system
        # First check which line is which
        if p0_l1[1] > p0_l0[1]:
            p0_tmp = p0_l0
            p0_l0 = p0_l1
            p0_l1 = p0_tmp
            p1_tmp = p1_l0
            p1_l0 = p1_l1
            p1_l1 = p1_tmp
            v_tmp = v_l0
            v_l0 = v_l1
            v_l1 = v_tmp


        right_to_center_direction = v_l1 - v_l0
        right_to_center_direction /= np.linalg.norm(right_to_center_direction)
        center_point = intersection_point_3D + 0.5 * np.sqrt(2) * right_to_center_direction

        x_axis = v_l0
        y_axis = v_l1
        z_axis = normal_vector

        #Visualization
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for i in range(0, filtered_pointcloud.shape[0], 500):
        #     point = filtered_pointcloud[i]
        #     ax.scatter(point[0], point[1], point[2], c='b', marker='o')
        # #ax.scatter(center_point[0], center_point[1], center_point[2], c='k', marker='o')
        # #ax.scatter(intersection_point_3D[0], intersection_point_3D[1], intersection_point_3D[2], c='r', marker='o')
        # #ax.quiver(p0_l1[0], p0_l1[1], p0_l1[2], v_l1[0], v_l1[1], v_l1[2])
        # #ax.quiver(p0_l0[0], p0_l0[1], p0_l0[2], v_l0[0], v_l0[1], v_l0[2])
        # ax.quiver(center_point[0], center_point[1], center_point[2], x_axis[0], x_axis[1], x_axis[2])
        # ax.quiver(center_point[0], center_point[1], center_point[2], y_axis[0], y_axis[1], y_axis[2])
        # #ax.quiver(closest_point_line0[0], closest_point_line0[1], closest_point_line0[2], normal_vector[0], normal_vector[1], normal_vector[2])
        # #ax.quiver(center_point[0], center_point[1], center_point[2], y_axis[0], y_axis[1], y_axis[2])
        # #ax.quiver(center_point[0], center_point[1], center_point[2], z_axis[0], z_axis[1], z_axis[2])
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()



        return center_point, x_axis, y_axis, z_axis

if __name__ == '__main__':
    # Imports for testing
    import pcl
    relative_utils_path = '../utils'
    utils_path = os.path.join(os.path.dirname(__file__), relative_utils_path)
    sys.path.append(utils_path)
    from pcl_helper import float_to_rgb
    # Test Code
    print('Running table recognizer...')
    relative_pointcloud_path = '../../data/pointclouds5/table10_1516127883476557.pcd'
    pointcloud_path = os.path.join(os.path.dirname(__file__), relative_pointcloud_path)
    pointcloud = pcl.load_XYZRGB(pointcloud_path)
    pointcloud_np = pointcloud.to_array()

    # Compute RGB image
    rgb_image = []
    for i in range(0, pointcloud_np.shape[0]):
        rgb_image.append(float_to_rgb(pointcloud_np[i, 3]))
    rgb_image_np = np.asarray(rgb_image, dtype=np.uint8)
    rgb_image_np = np.reshape(rgb_image_np, (480, 640, 3))

    table_recognizer = TableRecognizer()
    table_recognizer.initialize_recognizer()
    table_recognizer.predict_center_point(pointcloud_np, rgb_image_np)

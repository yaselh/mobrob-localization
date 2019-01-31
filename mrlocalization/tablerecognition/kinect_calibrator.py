import os
import sys
import numpy as np

import math
from math import radians

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tablerecognizerv2 import TableRecognizer

class KinectCalibrator:

    relative_settings_path = '../../settings'
    settings_path = os.path.join(os.path.dirname(__file__), relative_settings_path)

    def __init__(self):
        self.table_recognizer = TableRecognizer()



    def perform_calibration(self, pointcloud_np):
        print('Performing table recognition')

        # Compute RGB image
        rgb_image = []
        for i in range(0, pointcloud_np.shape[0]):
            rgb_image.append(float_to_rgb(pointcloud_np[i, 3]))
        rgb_image_np = np.asarray(rgb_image, dtype=np.uint8)
        rgb_image_np = np.reshape(rgb_image_np, (480, 640, 3))
        try:
            center_point, x_axis, y_axis, z_axis = self.table_recognizer.predict_center_point(pointcloud_np, rgb_image_np)
        except RuntimeError as e:
            print('Table could not be recognized from given frame.')
            return
        translation_vector = -center_point
        rotation_matrix = np.transpose(np.array([x_axis, y_axis, z_axis]))
        new_center_point = self.adjust_center_point(pointcloud_np[:,:3], center_point, translation_vector, rotation_matrix)
        new_rotation = self.adjust_y_rotation(pointcloud_np[:,:3], new_center_point, translation_vector, rotation_matrix)
        new_rotation = self.adjust_x_rotation(pointcloud_np[:,:3], new_center_point, translation_vector, rotation_matrix)
        new_rotation = self.adjust_z_rotation(pointcloud_np[:,:3], new_center_point, translation_vector, rotation_matrix)

        translation_vector = -new_center_point
        rotation_matrix = new_rotation
        save_string = input("Do you want to save? (y/n)")
        if save_string == 'y':
            np.save(os.path.join(KinectCalibrator.settings_path, 'translation.npy'), translation_vector)
            np.save(os.path.join(KinectCalibrator.settings_path, 'rotation.npy'), rotation_matrix)





    def adjust_y_rotation(self, pointcloud_np, center_point, translation_vector, rotation_matrix):
        # Filter all nan value points and points further than 2m
        #pointcloud_np = pointcloud_np[pointcloud_np[:,2] < 2.0]
        result_ok = False
        while not result_ok:
            pointcloud_np_new = np.copy(pointcloud_np)
            pointcloud_np_new = pointcloud_np_new[np.logical_not(np.isnan(pointcloud_np_new[:,2]))]
            pointcloud_np_new = pointcloud_np_new + translation_vector
            pointcloud_np_new = np.dot(pointcloud_np_new, rotation_matrix)
            #pointcloud_np = self.crop_from_scene(pointcloud_np, -1.0, 1.0, -1.0, 1.0, -0.045, 0.045, False)
            pointcloud_2d = pointcloud_np_new[:,[0,2]]
            fig, ax = self.render_pointcloud_2d(pointcloud_2d, 100)
            plt.show()
            result_ok_str = str(input('Is result okay (y/n)?'))
            if result_ok_str == 'y':
                result_ok = True
                continue
            else:
                result_ok = False
            degrees= float(input('Enter y-axis rotation:'))
            adjust_matrix = self.__rotation_matrix(np.array([0,1,0]), radians(degrees))
            rotation_matrix = np.dot(rotation_matrix, adjust_matrix)
        return rotation_matrix


    def adjust_x_rotation(self, pointcloud_np, center_point, translation_vector, rotation_matrix):
        # Filter all nan value points and points further than 2m
        #pointcloud_np = pointcloud_np[pointcloud_np[:,2] < 2.0]
        result_ok = False
        while not result_ok:
            pointcloud_np_new = np.copy(pointcloud_np)
            pointcloud_np_new = pointcloud_np_new[np.logical_not(np.isnan(pointcloud_np_new[:,2]))]
            pointcloud_np_new = pointcloud_np_new + translation_vector
            pointcloud_np_new = np.dot(pointcloud_np_new, rotation_matrix)
            #pointcloud_np = self.crop_from_scene(pointcloud_np, -1.0, 1.0, -1.0, 1.0, -0.045, 0.045, False)
            pointcloud_2d = pointcloud_np_new[:,[1,2]]
            fig, ax = self.render_pointcloud_2d(pointcloud_2d, 100)
            plt.show()
            result_ok_str = str(input('Is result okay (y/n)?'))
            if result_ok_str == 'y':
                result_ok = True
                continue
            else:
                result_ok = False
            degrees= float(input('Enter x-axis rotation:'))
            adjust_matrix = self.__rotation_matrix(np.array([1,0,0]), -radians(degrees))
            rotation_matrix = np.dot(rotation_matrix, adjust_matrix)
        return rotation_matrix

    def adjust_z_rotation(self, pointcloud_np, center_point, translation_vector, rotation_matrix):
        # Filter all nan value points and points further than 2m
        #pointcloud_np = pointcloud_np[pointcloud_np[:,2] < 2.0]
        result_ok = False
        while not result_ok:
            pointcloud_np_new = np.copy(pointcloud_np)
            pointcloud_np_new = pointcloud_np_new[np.logical_not(np.isnan(pointcloud_np_new[:,2]))]
            pointcloud_np_new = pointcloud_np_new + translation_vector
            pointcloud_np_new = np.dot(pointcloud_np_new, rotation_matrix)
            #pointcloud_np = self.crop_from_scene(pointcloud_np, -1.0, 1.0, -1.0, 1.0, -0.045, 0.045, False)
            pointcloud_2d = pointcloud_np_new[:,[0,1]]
            fig, ax = self.render_pointcloud_2d(pointcloud_2d, 100)
            plt.show()
            result_ok_str = str(input('Is result okay (y/n)?'))
            if result_ok_str == 'y':
                result_ok = True
                continue
            else:
                result_ok = False
            degrees= float(input('Enter z-axis rotation:'))
            adjust_matrix = self.__rotation_matrix(np.array([0,0,1]), -radians(degrees))
            rotation_matrix = np.dot(rotation_matrix, adjust_matrix)
        return rotation_matrix

    def adjust_center_point(self, pointcloud_np, center_point,translation_vector, rotation_matrix):
        # Filter all nan value points and points further than 2m
        pointcloud_np = np.copy(pointcloud_np)
        pointcloud_np = pointcloud_np[np.logical_not(np.isnan(pointcloud_np[:,2]))]
        pointcloud_np = pointcloud_np[pointcloud_np[:,2] < 2.0]
        pointcloud_np = pointcloud_np + translation_vector
        pointcloud_np = np.dot(pointcloud_np, rotation_matrix)
        pointcloud_np = self.crop_from_scene(pointcloud_np, -1.0, 1.0, -1.0, 1.0, -0.02, 0.02, True)
        pointcloud_2d = pointcloud_np[:,:2]
        fig, ax = self.render_pointcloud_2d(pointcloud_2d, 50)

        new_center = np.array([0.0, 0.0, 0.0])
        def onclick(event):
            new_center[0] = event.xdata
            new_center[1] = event.ydata
            plt.close()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        new_center_point = np.dot(new_center, np.transpose(rotation_matrix))
        new_center_point = new_center_point - translation_vector
        return new_center_point

    def render_pointcloud(self, pointcloud_np, step_width):
        # Visualization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(0, pointcloud_np.shape[0], step_width):
            point = pointcloud_np[i]
            ax.scatter(point[0], point[1], point[2], c='b', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        return fig, ax

    def render_pointcloud_2d(self, pointcloud_np, step_width):
        pointcloud_np = pointcloud_np[:,:2]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(0, pointcloud_np.shape[0], step_width):
            point = pointcloud_np[i]
            ax.scatter(point[0], point[1], marker='o', c='b')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        return fig, ax

    def crop_from_scene(self, pointcloud, minx, maxx, miny, maxy, minz, maxz, negative):
        is_in_xrange = np.logical_and(pointcloud[:,0] > minx,pointcloud[:,0] < maxx)
        is_in_yrange = np.logical_and(pointcloud[:,1] > miny,pointcloud[:,1] < maxy)
        is_in_zrange = np.logical_and(pointcloud[:,2] > minz,pointcloud[:,2] < maxz)
        is_in_all_ranges = np.logical_and(is_in_xrange, is_in_yrange)
        is_in_all_ranges = np.logical_and(is_in_all_ranges, is_in_zrange)
        if negative:
            outcloud = pointcloud[np.logical_not(is_in_all_ranges)]
        else:
            outcloud = pointcloud[is_in_all_ranges]
        return outcloud

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

if __name__=='__main__':
        relative_utils_path = '../utils'
        utils_path = os.path.join(os.path.dirname(__file__), relative_utils_path)
        sys.path.append(utils_path)
        from pcl_helper import float_to_rgb
        # Test Code
        print('Running kinect calibrator...')
        relative_pointcloud_path = '../../data/calibration_scenes/scene180130.npy'
        pointcloud_path = os.path.join(os.path.dirname(__file__), relative_pointcloud_path)
        pointcloud_np = np.load(pointcloud_path)

        calibrator = KinectCalibrator()
        calibrator.perform_calibration(pointcloud_np)

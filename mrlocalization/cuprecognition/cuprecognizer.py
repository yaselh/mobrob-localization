from __future__ import division
import os
import sys
import numpy as np


relative_svm_recognizer_dir = './svm_recognizer'
svm_recognizer_dir = os.path.join(os.path.dirname(__file__), relative_svm_recognizer_dir)
sys.path.append(svm_recognizer_dir)

relative_pose_estimator_dir = './pose_estimator'
pose_estimator_dir = os.path.join(os.path.dirname(__file__), relative_pose_estimator_dir)
sys.path.append(pose_estimator_dir)

relative_bbox_calc_dir = '../utils'
bbox_calc_dir = os.path.join(os.path.dirname(__file__), relative_bbox_calc_dir)
sys.path.append(bbox_calc_dir)

relative_msg_dir = '../msgs'
msg_dir = os.path.join(os.path.dirname(__file__), relative_msg_dir)
sys.path.append(msg_dir)

from svm_recognizer import SVMRecognizer
from bbox_calculator import minimum_bounding_rectangle
from cup_pose_estimator import CupPoseEstimator
from bbox import BBox

from sklearn.cluster import DBSCAN

import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

class CupRecognizer:

    def __init__(self):
        self.recognizer = SVMRecognizer() # Change this line to use different recognizer!!!!
        self.pose_estimator = CupPoseEstimator()
        self.last_filtered_bboxes = []

    def initialize_recognizer(self):
        self.recognizer.initialize_recognizer()

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

    def predict_bboxes(self, rgb_image_np, pointcloud_np, translation_vector, rotation_matrix):
        print('Predicting bboxes')


        # Recognize cups
        try:
            bbox_list = self.recognizer.recognize(rgb_image_np, pointcloud_np) # Use real recognition method here!
            print('Recognizer returned bboxes in image at {}.'.format(bbox_list))
        except RuntimeError as e:
            print(e)
            return [] # return empty bounding box list

        # If no cups present or too many cups present quit. Later maybe implement some decision mechanism.
        # if len(bbox_list) != 1:
        #     print('Error: Number of detected cups != 1. Detected {} cups.'.format(len(bbox_list)))
        #     return None

        # Quit if no bbox found
        if len(bbox_list) < 1:
            raise RuntimeError('No bounding box was extracted by the recognition component.')


        # Filter bboxes that are too far away from table
        pointcloud_image_shape_np = np.reshape(pointcloud_np, (480,640,4))
        center_point = np.array([0.0, 0.0, 0.0])
        filtered_bbox_list = []
        for bbox in bbox_list:
            bbox_size_percentage = int(min(bbox[2], bbox[3]) * 0.2)
            bbox_center_row = bbox[0] + bbox[2] // 2
            bbox_center_col = bbox[1] + bbox[3] // 2
            bbox_area_pointcloud = pointcloud_image_shape_np[bbox_center_row-bbox_size_percentage:bbox_center_row+bbox_size_percentage,bbox_center_col-bbox_size_percentage:bbox_center_col+bbox_size_percentage,:]
            median_values = np.nanmedian(bbox_area_pointcloud, axis=(0,1))[:3]
            median_values = median_values + translation_vector
            median_values = np.dot(median_values, rotation_matrix)
            if np.abs(median_values[2]) < 0.1:
                filtered_bbox_list.append(bbox)
        self.last_filtered_bboxes = filtered_bbox_list

        # Check again if bounding boxes are still available
        if len(filtered_bbox_list) < 1:
            raise RuntimeError('No bounding box has plausible pointcloud coordinates.')



        # Heuristic to select most centered bounding box
        most_centered_bbox = None
        most_centered_bbox_dist = sys.float_info.max
        image_center = np.array([int(rgb_image_np.shape[0] / 2), int(rgb_image_np.shape[1] / 2)])
        for bbox in filtered_bbox_list:
            bbox_center = np.array([int(bbox[0] + 0.5 * bbox[2]), int(bbox[1] + 0.5 * bbox[3])])
            bbox_dist = np.linalg.norm(bbox_center-image_center)
            if bbox_dist < most_centered_bbox_dist:
                most_centered_bbox = bbox
                most_centered_bbox_dist = bbox_dist
        print('Selected bbox at {}.'.format(most_centered_bbox))


        # Calculate cup center in image
        #bbox = bbox_list[0] # Treat first result as relevant bbox (Algorithm should have quit if numer of cups != 1)
        bbox = most_centered_bbox # Take the one that belongs to the most centered detection

        bbox_center_row = bbox[0] + bbox[2] // 2
        bbox_center_col = bbox[1] + bbox[3] // 2
        bbox_size_percentage = int(min(bbox[2], bbox[3]) * 0.2)

        # Calculate approximate center of cup in pointcloud
        cup_area_pointcloud = pointcloud_image_shape_np[bbox_center_row-bbox_size_percentage:bbox_center_row+bbox_size_percentage,bbox_center_col-bbox_size_percentage:bbox_center_col+bbox_size_percentage,:]
        # Calculates median by ignoring NANs which are present in pointcloud.
        #Probably raises All-Nan wartning which occurs because color values are somehow NAN. Not relevant here.
        median_values = np.nanmedian(cup_area_pointcloud, axis=(0,1))[:3]
        # Crop box around cup
        # Remove table plate from pointcloud
        pointcloud_no_table = np.copy(pointcloud_np[:,:3])
        pointcloud_no_table = pointcloud_no_table[np.logical_not(np.any(np.isnan(pointcloud_no_table), axis=1))]
        pointcloud_no_table = pointcloud_no_table[pointcloud_no_table[:,2] < 1.75]
        #pointcloud_no_table = pointcloud_no_table[]
        pointcloud_no_table = pointcloud_no_table + translation_vector
        pointcloud_no_table = np.dot(pointcloud_no_table, rotation_matrix)
        pointcloud_no_table = pointcloud_no_table[pointcloud_no_table[:,2] > -0.1]
        pointcloud_no_table = self.crop_from_scene(pointcloud_no_table, -1.0, 1.0, -1.0, 1.0, -0.05, 0.05, True)


        median_values = median_values + translation_vector
        median_values = np.dot(median_values, rotation_matrix)
        cup_pointcloud = self.crop_from_scene(pointcloud_no_table, median_values[0]-0.15, median_values[0]+0.15, median_values[1]-0.15, median_values[1]+0.15, median_values[2]-0.15, median_values[2]+0.15, False)

        if cup_pointcloud.shape[0] <= 0:
            print('Pointcloud of cup has length zero. Skipping bbox calculation.')
            return []
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for i in range(0,pointcloud_no_table.shape[0],100):
        #     point = pointcloud_no_table[i]
        #     ax.scatter(point[0], point[1], point[2], c='b', marker='o')
        # #ax.scatter(0, 0, 0, c='r')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()
        # visual = pcl.pcl_visualization.CloudViewing()
        # visual.ShowMonochromeCloud(cup_pointcloud, b'cloud')
        # time.sleep(20)

        # Cluster near together points to segment cup from outliers

        db = DBSCAN(eps=0.05, min_samples=50).fit(cup_pointcloud)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print('Clusters: {}'.format(n_clusters))


        #Visualize clustering result
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for i in range(0, n_clusters):
        #     if i==0:
        #         c='b'
        #     elif i==1:
        #         c='y'
        #     elif i==2:
        #         c='m'
        #     else:
        #         c='r'
        #     cluster_points = cup_pointcloud[labels==i]
        #     for i in range(0,len(cluster_points),2):
        #         point = cluster_points[i]
        #         ax.scatter(point[0], point[1], point[2], c=c, marker='o')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()

        # Select biggest cluster from pointcloud
        biggest_cluster_count = 0
        biggest_cluster = None
        for i in range(0, n_clusters):
            cluster_points = cup_pointcloud[labels==i]
            if cluster_points.shape[0] > biggest_cluster_count:
                biggest_cluster_count = cluster_points.shape[0]
                biggest_cluster = cluster_points

        if biggest_cluster is None or (biggest_cluster.shape[0] < 400 or biggest_cluster.shape[0] > 2000):
            print('Cluster is too big and is probably not the cup.')
            return [] # Return empty bounding box list

        print('Biggest cluster consists of {} points.'.format(biggest_cluster.shape[0]))

        bbox_list_3d = []
        cup_bbox = self.get_cluster_bbox(biggest_cluster)
        cup_bbox.object_class = 'cup'
        bbox_list_3d.append(cup_bbox)

        print('Recognized bbox at ({}, {}, {}).'.format(cup_bbox.x, cup_bbox.y, cup_bbox.z))

        # Estimate pose of cup
        handle_angle = self.pose_estimator.estimate_pose(biggest_cluster)
        cup_bbox.Y = handle_angle

        # # Estimate pose of cup
        # cup_cluster_table_coordinates_2d = cup_cluster_table_coordinates[:,:2]
        # cup_rect = minimum_bounding_rectangle(cup_cluster_table_coordinates_2d)
        # cup_rect_center = cup_rect[0] + 0.5 * (cup_rect[2] - cup_rect[0])
        # cup_average_center = np.average(cup_cluster_table_coordinates_2d, axis=0)
        #
        #
        # side1 = cup_rect[1] - cup_rect[0]
        # side1_length = np.linalg.norm(side1)
        # side2 = cup_rect[2] - cup_rect[1]
        # side2_length = np.linalg.norm(side2)
        # side3 = cup_rect[3] - cup_rect[2]
        # side3_length = side1_length
        # side4 = cup_rect[0] - cup_rect[3]
        # side4_length = side2_length
        #
        # side1_middle = cup_rect[0] + 0.5 * side1
        # side2_middle = cup_rect[1] + 0.5 * side2
        # side3_middle = cup_rect[2] + 0.5 * side3
        # side4_middle = cup_rect[3] + 0.5 * side4
        #
        # eps = 0.01
        # if isclose(side1_length, side2_length, abs_tol=0.01):
        #     print('Handle was not clearly identified. Assuming y axis direction.')
        #     handle_direction = np.array([0,1,0])
        # elif side1_length > side2_length:
        #     if np.linalg.norm(cup_average_center - side2_middle) > np.linalg.norm(cup_average_center - side4_middle):
        #         handle_direction = side1
        #     else:
        #         handle_direction = -side1
        # else:
        #     if np.linalg.norm(cup_average_center - side1_middle) > np.linalg.norm(cup_average_center - side3_middle):
        #         handle_direction = side2
        #     else:
        #         handle_direction = -side2
        # yaw_angle_y_axis = math.atan2(handle_direction[1], handle_direction[0]) - math.atan2(1, 0);
        # print('Cup handle angle to y axis is {}.'.format(yaw_angle_y_axis))
        #
        # Visualize cup in table coordinate system in 2D
        # cup_cluster_table_coordinates = biggest_cluster
        # fig = plt.figure(figsize=(8,8))
        # ax = fig.add_subplot(111)
        # for i in range(0,len(cup_cluster_table_coordinates),1):
        #     point = cup_cluster_table_coordinates[i]
        #     ax.scatter(point[0], point[1], c='b', marker='o')
        #ax.scatter(cup_average_center[0], cup_average_center[1], c='m', marker='o')
        #ax.quiver(cup_average_center[0], cup_average_center[1], handle_direction[0], handle_direction[1])
        # ax.plot([-0.5, 0.5], [0.5, 0.5], color='k', linestyle='-', linewidth=2)
        # ax.plot([0.5, 0.5], [0.5, -0.5], color='k', linestyle='-', linewidth=2)
        # ax.plot([0.5, -0.5], [-0.5, -0.5], color='k', linestyle='-', linewidth=2)
        # ax.plot([-0.5, -0.5], [-0.5, 0.5], color='k', linestyle='-', linewidth=2)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_xlim(-0.5,0.5)
        # ax.set_ylim(-0.5,0.5)
        # plt.show()
        return bbox_list_3d

    def get_cluster_bbox(self, cluster):
        print('Calculating bbox dimensions...')
        bbox = BBox()
        max_values = np.amax(cluster, axis=0)
        min_values = np.amin(cluster, axis=0)
        mean = np.mean(cluster, axis=0)
        bbox.x = mean[0]
        bbox.y = mean[1]
        bbox.z = max_values[2] - max_values[2] / 2

        x_diff = abs(max_values[0] - min_values[0])
        y_diff = abs(max_values[1] - min_values[1])
        max_side = max(x_diff, y_diff)
        bbox.width = max_side
        bbox.length = max_side
        bbox.height = max_values[2] - min_values[2]
        return bbox



if __name__=='__main__':
    # Imports for testing
    relative_utils_path = '../utils'
    utils_path = os.path.join(os.path.dirname(__file__), relative_utils_path)
    sys.path.append(utils_path)
    from pcl_helper import float_to_rgb

    # Load pointcloud
    print('Running CupRecognizer...')
    # Good table71514984110759315.pcd
    # Problem table61514984092239338.pcd
    relative_pointcloud_path = '../../data/pointclouds7/1517154270.npy'
    pointcloud_path = os.path.join(os.path.dirname(__file__), relative_pointcloud_path)
    pointcloud_np = np.load(pointcloud_path)

    # Compute RGB image
    rgb_image = []
    for i in range(0, pointcloud_np.shape[0]):
        rgb_image.append(float_to_rgb(pointcloud_np[i, 3]))
    rgb_image_np = np.asarray(rgb_image, dtype=np.uint8)
    rgb_image_np = np.reshape(rgb_image_np, (480, 640, 3))

    plt.imshow(rgb_image_np)
    plt.show()

    # Recognize table
    relative_settings_path = '../../settings'
    settings_path = os.path.join(os.path.dirname(__file__),relative_settings_path)
    translation_vector = np.load(os.path.join(settings_path, 'translation.npy'))
    rotation_matrix = np.load(os.path.join(settings_path, 'rotation.npy'))

    # Transform pointcloud in other coordinate system
    # filtered_pointcloud_np = filtered_pointcloud.to_array()
    # pointcloud_new_origin = filtered_pointcloud_np[:,:3] - table_center_point
    # pointcloud_final = np.dot(pointcloud_new_origin, transform_matrix)


    # Transformation visualization
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(0,len(pointcloud_final),100):
    #     point = pointcloud_final[i]
    #     ax.scatter(point[0], point[1], point[2], c='b', marker='o')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()



    #visual = pcl.pcl_visualization.CloudViewing()
    #visual.ShowColorCloud(pointcloud, b'cloud')
    #flag = True
    #while flag:
    #    flag != visual.WasStopped()
    #end

    cup_recognizer = CupRecognizer()
    cup_recognizer.initialize_recognizer()
    cup_recognizer.predict_bboxes(rgb_image_np, pointcloud_np, translation_vector, rotation_matrix)

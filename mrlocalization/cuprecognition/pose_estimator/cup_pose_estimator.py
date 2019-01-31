from __future__ import division
import numpy as np

from math import degrees, pi

from sklearn.linear_model import RANSACRegressor
from cup_handle_base_estimator import CupHandleBaseEstimator
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN


import matplotlib.pyplot as plt

class CupPoseEstimator:

    def estimate_pose(self, cup_pointcloud):
        print('Estimating Pose...')
        # Reduce to 2D
        cup_pointcloud_2d = cup_pointcloud[:,:2]

        # Find outliers
        n_neighbors = int(cup_pointcloud_2d.shape[0] * 0.9)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, n_jobs=-1)
        outlier_detection_result = lof.fit_predict(cup_pointcloud_2d)
        outliers = cup_pointcloud_2d[outlier_detection_result==-1]
        non_outliers = cup_pointcloud_2d[outlier_detection_result==1]

        if outliers.shape[0] == 0:
            return 0.0

        centroid_2d = np.mean(non_outliers ,axis=0)

        db = DBSCAN(eps=0.02, min_samples=10).fit(outliers)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        selected_cluster = None
        selected_cluster_average = None
        max_distance = 0
        min_distance = 999.0
        for i in range(0, n_clusters):
            cluster = outliers[labels == i]
            average = np.average(cluster, axis = 0)
            distance = np.linalg.norm(average - centroid_2d)
            if distance > max_distance:
                max_distance = distance
                selected_cluster = cluster
                selected_cluster_average = average
            if distance < min_distance:
                min_distance = distance

        distance_difference = max_distance - min_distance

        if selected_cluster.shape[0] > 20 and max_distance > 0.04 or max_distance - min_distance > 0.015:
            handle_vector = selected_cluster_average - centroid_2d
            y_axis = np.array([0.0, 1.0])
            angle = self.angle_between(y_axis, handle_vector)
        else:
            angle = 0.0

        print('Recognized angle: {} degrees.'.format(degrees(angle)))
        #Outlier detection visualization
        # for i, point in enumerate(cup_pointcloud_2d):
        #     if outlier_detection_result[i] == -1:
        #         plt.scatter(point[0], point[1], c='r')
        #     else:
        #         plt.scatter(point[0], point[1], c='b')
        # plt.scatter(selected_cluster_average[0], selected_cluster_average[1], c='k')
        # plt.scatter(centroid_2d[0], centroid_2d[1], c='g')
        # plt.show()


        return angle

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        angle = np.arctan2(v2_u[1], v2_u[0]) - np.arctan2(v1_u[1], v1_u[0])
        if angle < 0:
            return 2*pi + angle
        else:
            return angle
        return np.arctan2(v2_u[1], v2_u[0]) - np.arctan2(v1_u[1], v1_u[0])

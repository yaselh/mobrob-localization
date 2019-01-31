from __future__ import division

import os
import sys

import numpy as np
import time

from sklearn.cluster import DBSCAN

import skimage
from skimage.color import rgb2gray
import skimage.io as io

predictor_relative_path = '../../deepnn/models/svm'
predictor_path = os.path.join(os.path.dirname(__file__), predictor_relative_path)
sys.path.append(predictor_path)

from svm_predictor import SVMPredictor

from multiprocessing.pool import ThreadPool

import matplotlib
import matplotlib.pyplot as plt

class SVMRecognizer:

    NUM_WORKERS = 8

    def initialize_recognizer(self):
        self.predictor = SVMPredictor()
        self.predictor.initialize_predictor()
        self.threadpool = ThreadPool(SVMRecognizer.NUM_WORKERS)


    def resize_image(self, image, max_side_length):
        image_height = image.shape[0]
        image_width = image.shape[1]
        ratio = min(max_side_length / image_height, max_side_length / image_width);
        image = skimage.transform.resize(image, (int(image_height * ratio), int(image_width * ratio)), mode='constant')
        return image

    def get_sliding_window_patches(self, image, height=50, width=50, stride=5, start_row=0, start_col=0, end_row=-1, end_col=-1):
        row_count = image.shape[0]
        col_count = image.shape[1]
        patches = [] # Format (row, col, height, width)
        if end_row  == -1:
            end_row = row_count
        if end_col == -1:
            end_col = col_count

        for col in range(start_col, (end_col - width) + 1, stride):
            for row in range(start_row, (end_row - height) + 1, stride):
                patch = image[row:row+height, col:col+width]
                patches.append((row, col, height, width, patch))
        return patches

    # found at https://stackoverflow.com/questions/752308/split-list-into-smaller-lists
    def split_list(self, alist, wanted_parts=1):
        length = len(alist)
        return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
                for i in range(wanted_parts) ]

    def predict_multithreaded(self, patches):
        num_threads = SVMRecognizer.NUM_WORKERS
        chunks = self.split_list(patches, wanted_parts=num_threads)
        results = self.threadpool.map(self.predictor.predict, chunks)
        overall_results = []
        for result in results:
            overall_results.extend(result)
        return overall_results

    def recognize(self, image, pointcloud):
        grayscale_image = rgb2gray(image)

        original_image_height = grayscale_image.shape[0]
        original_image_width = grayscale_image.shape[1]
        image_levels = [600]

        overall_results = []
        overall_bboxes = []
        for image_level in image_levels:
            current_image = self.resize_image(grayscale_image, image_level)
            if current_image.shape[0] < 70 or current_image.shape[1] < 70:
                break
            patches_with_pos = self.get_sliding_window_patches(current_image, 70, 70, 10)
            patches = []
            for patch in patches_with_pos:
                patches.append(patch[4])

            start_time = time.time()
            result = self.predict_multithreaded(patches)
            end_time = time.time()
            duration = end_time - start_time
            print('Prediction took {}s'.format(duration))

            # Append results for this level to overall result array
            overall_results.extend(result)
            # Append bounding box positions and sizes to overall bbox array
            current_image_height = current_image.shape[0]
            height_ratio = original_image_height / current_image_height
            current_image_width = current_image.shape[1]
            width_ratio = original_image_width / current_image_width
            for patch in patches_with_pos:
                original_patch = (int(patch[0] * height_ratio), int(patch[1] * width_ratio), int(patch[2] * height_ratio), int(patch[3] * width_ratio))
                overall_bboxes.append(original_patch)



            # Get boundingboxes with high enough values
            overall_bboxes_np = np.asarray(overall_bboxes)
            overall_results_np = np.asarray(overall_results)

            condition = overall_results_np[:,1] > 0.6 #Set threshold here
            good_results = overall_results_np[condition]
            good_bboxes = overall_bboxes_np[condition]

            # Calculate bbox centers
            good_bbox_centers = []
            for bbox in good_bboxes:
                good_bbox_centers.append([int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)])
            good_bbox_centers_np = np.asarray(good_bbox_centers)

            # TODO Probably return here if no good bboxes present.
            if good_bboxes.shape[0] <= 0:
                raise RuntimeError('No bounding boxes identified in current image.')

            # Cluster near together boundingboxes
            db = DBSCAN(eps=20.0, min_samples=1).fit(good_bbox_centers_np)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            combined_bboxes = []
            for i in range(0, n_clusters):
                current_label_bboxes = good_bboxes[labels==i]
                averages = np.average(current_label_bboxes, axis=0)
                average_row = averages[0]
                average_col = averages[1]
                average_height = averages[2]
                average_width = averages[3]
                combined_bboxes.append((int(average_row), int(average_col), int(average_height), int(average_width)))


            #Visualization
            #Result evaluation
            #fig,ax = plt.subplots(1)
            # Display the image
            #ax.imshow(image)

            # for i in range(0, len(combined_bboxes)):
            #     bbox = combined_bboxes[i]
            #     sub_image =  image[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3],:]
            #     io.imsave('pos_classified_patches/{}_{}.png'.format(i,time.time()), sub_image)
            #    rect = matplotlib.patches.Rectangle((bbox[1],bbox[0]),bbox[3],bbox[2],linewidth=1,edgecolor='b',facecolor='none')
            #    ax.add_patch(rect)
            #plt.show()

            # Some Code for hard negative mining...
            #sub_image =  image[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3],:]
            #io.imsave('pos_classified_patches/{}_{}.png'.format(i,time.time()), sub_image)

        return combined_bboxes

if __name__=='__main__':
    # Imports for testing
    import pcl
    relative_utils_path = '../../utils'
    utils_path = os.path.join(os.path.dirname(__file__), relative_utils_path)
    sys.path.append(utils_path)
    from pcl_helper import float_to_rgb
    print('Running SVMRecognizer...')
    # Load pointcloud
    relative_pointcloud_path = '../../../data/pointclouds2/table11514983750201493.pcd'
    pointcloud_path = os.path.join(os.path.dirname(__file__), relative_pointcloud_path)
    pointcloud = pcl.load_XYZRGB(pointcloud_path)
    pointcloud_np = pointcloud.to_array()

    # Compute RGB image
    rgb_image = []
    for i in range(0, pointcloud_np.shape[0]):
        rgb_image.append(float_to_rgb(pointcloud_np[i, 3]))
    rgb_image_np = np.asarray(rgb_image, dtype=np.uint8)
    rgb_image_np = np.reshape(rgb_image_np, (480, 640, 3))

    # Create object and start recognition
    recognizer = SVMRecognizer()
    recognizer.initialize_recognizer()
    recognition_result = recognizer.recognize(rgb_image_np, pointcloud)

import os
import sys

import numpy as np
import cv2
from skimage import img_as_ubyte
from skimage.color import rgb2gray


import matplotlib.pyplot as plt
import matplotlib.patches as patches

predictor_relative_path = '../../deepnn/models/svm'
predictor_path = os.path.join(os.path.dirname(__file__), predictor_relative_path)
sys.path.append(predictor_path)
from svm_predictor import SVMPredictor

import time

class IPSVMRecognizer:

    def initialize_recognizer(self):
        self.predictor = SVMPredictor()
        self.predictor.initialize_predictor()


    def find_rois(self, image, fast_mode = True):
        image_start_row = 25
        image_stop_row = 479
        image_start_col = 50
        image_stop_col = 630
        original_image = np.copy(image)
        image = image[image_start_row:image_stop_row,image_start_col:image_stop_col,:]
        image = np.flip(image, axis = 2)
        image = cv2.resize(image, (0,0), fx=0.45, fy=0.45)
        cv2.setUseOptimized(True);
        cv2.setNumThreads(4);
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img_as_ubyte(image))

        if fast_mode:
            ss.switchToSelectiveSearchFast()
        else:
            ss.switchToSelectiveSearchQuality()
        rects = ss.process()

        row_factor = original_image.shape[0] / image.shape[0]
        col_factor = original_image.shape[1] / image.shape[1]
        rois = []
        for i in range(0, len(rects)):
            rect = rects[i]
            col = rect[0]
            row = rect[1]
            width = rect[3]
            height = rect[2]
            if height < 50 and height > 15 and width < 50 and width > 15:
                original_col = col * col_factor
                original_row = row * row_factor
                original_width = width * col_factor
                original_height = height * row_factor
                original_bbox = (int(original_row), int(original_col), int(original_height), int(original_width))
                rois.append(original_bbox)
        return rois

    def recognize(self, image, pointcloud):
        grayscale_image = rgb2gray(image)
        rois = self.find_rois(image, fast_mode=True)
        patches = []
        for roi in rois:
            row = roi[0]
            col = roi[1]
            height = roi[2]
            width = roi[3]
            patch = grayscale_image[row:row+height, col:col+width]
            patches.append(patch)
        results = self.predictor.predict(patches)
        for result, patch in zip(results, patches):
            if result[1] > 0.55:
                plt.imshow(patch, cmap='gray')
                plt.show()
        print(results)

if __name__=='__main__':
    # Imports for testing
    import pcl
    relative_utils_path = '../../utils'
    utils_path = os.path.join(os.path.dirname(__file__), relative_utils_path)
    sys.path.append(utils_path)
    from pcl_helper import float_to_rgb
    print('Running IPSVMRecognizer...')
    # Load pointcloud
    relative_pointcloud_path = '../../../data/pointclouds5/table9_1516127869728535.pcd'
    pointcloud_path = os.path.join(os.path.dirname(__file__), relative_pointcloud_path)
    pointcloud = pcl.load_XYZRGB(pointcloud_path)
    pointcloud_np = pointcloud.to_array()

    # Compute RGB image
    rgb_image = []
    for i in range(0, pointcloud_np.shape[0]):
        rgb_image.append(float_to_rgb(pointcloud_np[i, 3]))
    rgb_image_np = np.asarray(rgb_image, dtype=np.uint8)
    rgb_image_np = np.reshape(rgb_image_np, (480, 640, 3))

    # plt.imshow(rgb_image_np)
    # plt.show()

    # Create object and start recognition
    recognizer = IPSVMRecognizer()
    recognizer.initialize_recognizer()
    recognition_result = recognizer.recognize(rgb_image_np, pointcloud)

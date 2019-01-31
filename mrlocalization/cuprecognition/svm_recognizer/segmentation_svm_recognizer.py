import os
import sys

import numpy as np

from skimage.util import img_as_float
from skimage.color import rgb2gray
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.filters import sobel, gaussian
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops


import matplotlib.pyplot as plt

predictor_relative_path = '../../deepnn/models/svm'
predictor_path = os.path.join(os.path.dirname(__file__), predictor_relative_path)
sys.path.append(predictor_path)
from svm_predictor import SVMPredictor

class IPSVMRecognizer:

    def initialize_recognizer(self):
        self.predictor = SVMPredictor()
        self.predictor.initialize_predictor()


    def find_rois(self, image):
        image_start_row = 25
        image_stop_row = 479
        image_start_col = 50
        image_stop_col = 630
        original_image = np.copy(image)
        image = image[image_start_row:image_stop_row,image_start_col:image_stop_col,:]
        image = img_as_float(image)
        #image = gaussian(image, sigma=1, multichannel = True)

        grayscale_image = rgb2gray(image)
        gradient = sobel(grayscale_image)
        watershed_mask = np.ones(grayscale_image.shape, dtype=np.bool)
        segments_watershed = watershed(gradient, markers=250, compactness=0.0, mask=watershed_mask)

        valid_regions = []
        for region in regionprops(segments_watershed):
            if region.area > 50 and region.extent > 0.5:
                valid_regions.append(region)

        bbox_row_size = 70
        bbox_col_size = 70
        rois = []
        for region in valid_regions:
            centroid = region.centroid
            start_row = max(0, ((int(centroid[0]) - bbox_row_size // 2) + image_start_row))
            start_col = max(0, ((int(centroid[1]) - bbox_col_size // 2) + image_start_col))
            bbox = np.array([start_row, start_col, bbox_row_size, bbox_col_size])
            rois.append(bbox)

        plt.imshow(mark_boundaries(image, segments_watershed))
        for region in valid_regions:
            plt.scatter(region.centroid[1], region.centroid[0])
        plt.show()
        return rois

    def recognize(self, image, pointcloud):
        grayscale_image = rgb2gray(image)
        rois = self.find_rois(image)
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
            if result[1] > 0.5:
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
    relative_pointcloud_path = '../../../data/pointclouds5/table7_1516127824644917.pcd'
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

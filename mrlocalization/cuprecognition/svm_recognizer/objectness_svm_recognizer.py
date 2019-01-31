import os
import sys

import numpy as np

from skimage.color import rgb2gray, rgb2hsv
from skimage.util import img_as_float, img_as_ubyte
import skimage.io as io
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches

predictor_relative_path = '../../deepnn/models/svm'
predictor_path = os.path.join(os.path.dirname(__file__), predictor_relative_path)
sys.path.append(predictor_path)
from svm_predictor import SVMPredictor

bing_model_relative_path = '../../../models/BING/ObjectnessTrainedModel'
bing_model_path = os.path.join(os.path.dirname(__file__), bing_model_relative_path)

class IPSVMRecognizer:

    def initialize_recognizer(self):
        self.predictor = SVMPredictor()
        self.predictor.initialize_predictor()


    def find_rois(self, image):
        print(bing_model_path)
        image_start_row = 25
        image_stop_row = 479
        image_start_col = 50
        image_stop_col = 630
        #image = io.imread('C:/Users/Daniel/Code/python/mobrob-localization/data/processed/cupv2/val_images/000000562843.jpg')
        original_image = np.copy(image)
        #image = image[image_start_row:image_stop_row,image_start_col:image_stop_col,:]
        original_image = np.copy(image)
        image = rgb2hsv(image)
        image = img_as_ubyte(image)
        image = np.flip(image, axis = 2)
        print(image.shape)
        #image = gaussian(image, sigma=1, multichannel = True)
        bing = cv2.saliency.ObjectnessBING_create()
        bing.setTrainingPath(bing_model_path)
        bboxes = bing.computeSaliency(image)[1]
        objectness = bing.getobjectnessValues()
        print(objectness)
        print(len(objectness))
        fig,ax = plt.subplots(1)
        ax.imshow(original_image)
        for i in range(len(objectness) - 10050, len(objectness)-10000):
            print(objectness[i])
            bbox = bboxes[i][0]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            #if height < 100 and width < 100 and height > 60 and width > 60:
            #ax.scatter(bbox[0] + width / 2, bbox[1] + height / 2)
            rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
        plt.show()
        rois = []
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
    relative_pointcloud_path = '../../../data/pointclouds5/table9_1516127870160776.pcd'
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

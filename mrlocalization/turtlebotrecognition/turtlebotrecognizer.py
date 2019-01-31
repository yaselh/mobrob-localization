from __future__ import division
import os
import sys
import cv2
import time
import math
import numpy as np
from enum import Enum
import skimage.io as io
from skimage.color import rgb2gray
from sklearn.feature_extraction import image
from skimage.feature import match_template
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, erosion
from skimage import morphology
from skimage.filters import sobel
from scipy import ndimage as ndi
from skimage import feature
from skimage.color import label2rgb
from svm.svm_predictor import SVMPredictor
from matplotlib import pyplot as plt
from multiprocessing.pool import ThreadPool


relative_utils_path = '../utils'
utils_path = os.path.join(os.path.dirname(__file__), relative_utils_path)
sys.path.append(utils_path)

from pcl_helper import float_to_rgb

import os
import sys

relative_bbox_dir = '../msgs'
bbox_dir = os.path.join(os.path.dirname(__file__), relative_bbox_dir)
sys.path.append(bbox_dir)

from bbox import BBox

class Prediction_Method(Enum):
    TM=1
    SVM=2

class TurtlebotRecognizer:

    def __init__(self, template_path, method=Prediction_Method.TM, nbChunks=8):
        self.method = method
        self.template_path = template_path
	self.bbox = BBox()
	self.predictor = SVMPredictor()
	self.nbChunks = nbChunks
	self.threadpool = ThreadPool(self.nbChunks)


    def get_chunks(self, patches, num_chunks):
        length = len(patches)
        return [ patches[i*length // num_chunks: (i+1)*length // num_chunks]
                for i in range(num_chunks) ]


    def predict_parallel(self, patches):
        num_threads = self.nbChunks
        chunks = self.get_chunks(patches, num_threads)
        results = self.threadpool.map(self.predictor.predict, chunks)
        predictions = []
        for result in results:
            predictions.extend(result)
        return predictions




    def naive_match_template(self, frame_path, point_cloud=None):
        img_rgb = cv2.imread(frame_path)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        template = cv2.imread(self.template_path, 0)
        template = cv2.resize(template,None,fx=1, fy=1, interpolation = cv2.INTER_CUBIC)

        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv2.imwrite('res.png',img_rgb)

    #scale and rotation invariant template matching
    def match_template(self, frame_path, template_path):

        img1, img2 = frame_path, template_path

        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)

        kp2, des2 = orb.detectAndCompute(img2,None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # Extract the matched keypoints
        src_pts  = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts  = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        # Draw first 20 matches.
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20], flags=2,
                               outImg=None)
        plt.imshow(img3)
        plt.show()

        return src_pts, dst_pts

    def predict_bounding_box(self, frame, pointcloud_np, translation_vector, rotation_matrix, visualize=False):
        if self.method == Prediction_Method.TM:
            return self.predict_bounding_box_with_tm(frame, pointcloud_np, translation_vector, rotation_matrix, visualize)
        else:
            return self.predict_bounding_box_with_svm(frame, pointcloud_np, translation_vector, rotation_matrix, visualize)

    def predict_bounding_box_with_tm(self, frame_path, template_path, visualize=False):

        img1 = cv2.imread(template_path,0) # Turtlebot template
        img2 = cv2.imread(frame_path,0)     # frame from Kinect

        #crop frame
        x,y,h,w = 0,300,400,200
        #img2 = img2[y:y+h, x:x+w]

        #match the template and to the turtlebot in the given image
        src_pts, dst_pts = self.match_template(img1, img2)

        # Calculate homography matrix and perform perspective transformation
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        h,w = img1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        # Draw the bounding polygone
        img2 = cv2.polylines(img2, [np.int32(dst)], True, (255,0,0), 1, cv2.LINE_AA)

        plt.imshow(img2)
        plt.show()

        return dst

    def predict_bounding_box_with_svm(self, frame, pointcloud_np, translation_vector, rotation_matrix, visualize=False):
	#visualize = True
        # read the Kinect frame
        #img = io.imread(frame_path, as_grey=True)
	img = rgb2gray(frame)
	full_frame = rgb2gray(frame)
	_x,_y,_h,_w = 0,190,290,280
        img = img[_y:_y+_h, _x:_x+_w]
        # generate some patches from the frame
        h,w = (120, 120)
        patches = image.extract_patches_2d(img, (h,w), max_patches=400)

        # set the threshold
        threshold = 0.85
        # set the precision interval
        precision = 0

        # predict
	start_time = time.time()
        #predictions = self.predictor.predict(patches)
	predictions = self.predict_parallel(patches)
        end_time = time.time()
        duration = end_time - start_time
	#print "turtlebot inference time: " + str(duration)
        mean_x = []
        mean_y = []

        positive = 0
        for patch, prediction in zip(patches, predictions):
            if np.argmax(prediction) == 1 and prediction[1] >= threshold:

                # Match the patch of the turtlebot in the given image
                result = match_template(full_frame, patch)

                # -----------
                ij = np.unravel_index(np.argmax(result), result.shape)
                mean_x.append(ij[::-1][0])
                mean_y.append(ij[::-1][1])
                x, y = ij[::-1]
                positive += 1

        if positive != 0:
            x, y = np.mean(mean_x)+precision, np.mean(mean_y)+precision
            h, w = h-precision, w-precision


    	    h_in_frame = min(int(y+h), full_frame.shape[0])
    	    w_in_frame = min(int(x+w), full_frame.shape[1])

    	    # calculate the xyz-color matrix
    	    pointcloud_image_shape_np = np.reshape(pointcloud_np, (480,640,4))

            """
    	    cluster = []
    	    # estimate the depth where the turtlebot could be
    	    depth_map = pointcloud_image_shape_np[:,:,2]
    	    for i in range(int(y), h_in_frame):
    	    	for j in range(int(x), w_in_frame):
    			if not math.isnan(depth_map[i,j]):
    				if depth_map[i,j] > 1.50 and depth_map[i,j] < 1.90:
    					full_frame[i,j] = 0
    					cluster.append([i,j])
            """
    	    pred_turtlebot = full_frame[int(y):h_in_frame, int(x):w_in_frame]*255
    	    pred_turtlebot = closing(pred_turtlebot,square(3))
            """
    	    cluster_medians = np.nanmedian(cluster, axis=0)

            #cluster_center_row = int(cluster_medians[0])
            #cluster_center_col = int(cluster_medians[1])
            """
            #bbox_center_row = int(cluster_center_row + h // 2)
            #bbox_center_col = int(cluster_center_col + w // 2)
            bbox_center_row = int(y + h // 2)
            bbox_center_col = int(x + w // 2)

            """
            #match the template and to the turtlebot in the given image
            cv2.imwrite('bot.jpg', pred_turtlebot)
            bot = cv2.imread('bot.jpg')
            template = cv2.imread('templates/turtlebot_small.png', 0)
            src_pts, dst_pts = self.match_template(bot, template)
            # Calculate homography matrix and perform perspective transformation
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            _h,_w = bot.shape[:2]
            pts = np.float32([ [0,0],[0,_h-1],[_w-1,_h-1],[_w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            """

            if visualize:

                best_prediction = plt.Rectangle((x, y), h,  w,
                                                 edgecolor='r', facecolor='none')

                """
                # Draw the bounding polygone
                img = cv2.imread(frame_path, 0)
                img = cv2.polylines(bot, [np.int32(dst)], True, (255,0,0), 1, cv2.LINE_AA)
                """

                fig1 = plt.figure()
                ax1 = fig1.add_subplot(121, aspect='equal')
                ax2 = fig1.add_subplot(122, aspect='equal')
                #ax1.scatter(x=[cluster_center_col], y=[cluster_center_row], c='r')
                ax1.add_patch(best_prediction)
                ax1.imshow(full_frame)
                ax2.imshow(pred_turtlebot)
                plt.show()

            bbox_size_percentage = int(min(h, w) * 0.2)
            turtlebot_area_pointcloud = pointcloud_image_shape_np[bbox_center_row-bbox_size_percentage:bbox_center_row+bbox_size_percentage,bbox_center_col-bbox_size_percentage:bbox_center_col+bbox_size_percentage,:]

            median_values = np.nanmedian(turtlebot_area_pointcloud, axis=(0,1))[:3]
            median_values = median_values + translation_vector
            median_values = np.dot(median_values, rotation_matrix)


            self.bbox.object_class = "turtlebot"
            self.bbox.height = 0.41
            self.bbox.width = 0.34
            self.bbox.length = 0.29
            self.bbox.x = median_values[0]
            self.bbox.y = median_values[1]
            self.bbox.z = median_values[2] + self.bbox.height / 2
    	    #self.bbox.x = -0.8
    	    #self.bbox.y = 0.13
    	    #self.bbox.z = 0.0

            return self.bbox

if __name__ == "__main__":

    recognizer = TurtlebotRecognizer(None, method=Prediction_Method.TM)

    print recognizer.predict_bounding_box_with_tm('/home/yassinel/frame0000.png', 'templates/turtlebot_1.png')
    #print recognizer.predict_bounding_box_with_tm('/home/yassinel/test/77.png')

    import sys
    sys.exit()
    recognizer = TurtlebotRecognizer(None, method=Prediction_Method.SVM)
    #print recognizer.predict_bounding_box('/home/yassinel/test/1.png')
    #print recognizer.predict_bounding_box('/home/yassinel/test/77.png')

    translation_vector = np.load('../../settings/translation.npy')
    rotation_matrix = np.load('../../settings/rotation.npy')


    import glob
    pcs = glob.glob('/home/yassinel/tb_pcl/*.npy')
    for pointcloud_np in pcs:
    	pointcloud_np = np.load(pointcloud_np)
    	# Create RGB image from pointcloud
    	rgb_image = []
    	for i in range(0, pointcloud_np.shape[0]):
    		rgb_image.append(float_to_rgb(pointcloud_np[i, 3]))
    	rgb_image_np = np.asarray(rgb_image, dtype=np.uint8)
    	rgb_image_np = np.reshape(rgb_image_np, (480, 640, 3))

    	bbox = recognizer.predict_bounding_box(rgb_image_np,pointcloud_np, translation_vector, rotation_matrix, visualize=True)

    	print "x=" + str(bbox.x)
    	print "y=" + str(bbox.y)
    	print "z=" + str(bbox.z)
    	print "height=" + str(bbox.height)
    	print "width=" + str(bbox.width)
    	print "length=" + str(bbox.length)

# The main component that manages the localization process
from __future__ import division
import os
import sys
import numpy as np

import Queue
import threading
import time

import calendar
import time

from cup_acceptor_msgs.msg import BBox

relative_utils_path = './utils'
utils_path = os.path.join(os.path.dirname(__file__), relative_utils_path)
sys.path.append(utils_path)
from pcl_helper import float_to_rgb

relative_cuprecognition_path = './cuprecognition'
cuprecognition_path = os.path.join(os.path.dirname(__file__), relative_cuprecognition_path)
sys.path.append(cuprecognition_path)
from cuprecognizer import CupRecognizer

relative_turtlebotrecognition_path = './turtlebotrecognition'
turtlebotrecognition_path = os.path.join(os.path.dirname(__file__), relative_turtlebotrecognition_path)
sys.path.append(turtlebotrecognition_path)
from turtlebotrecognizer import *


class Localizer:

    MAX_QUEUE_SIZE = 1
    RELATIVE_SETTINGS_PATH = '../settings'
    SETTINGS_PATH = os.path.join(os.path.dirname(__file__), RELATIVE_SETTINGS_PATH)

    RELATIVE_DATA_LOG_PATH = '../data/log'
    DATA_LOG_PATH = os.path.join(os.path.dirname(__file__), RELATIVE_DATA_LOG_PATH)
    LOG_INPUT = False


    def __init__(self):
        self.queue = Queue.Queue(self.MAX_QUEUE_SIZE)
        self.thread = None
        self.stop_event = threading.Event()
        self.stop_event.set()
        self.initialized = False
        self.observers = []

    def initialize_localizer(self):

        self.cuprecognizer = CupRecognizer()
        self.cuprecognizer.initialize_recognizer()
        self.turtlebotrecognizer = TurtlebotRecognizer(None, method=Prediction_Method.SVM)
        self.initialized = True
        self.translation_vector = np.load(os.path.join(Localizer.SETTINGS_PATH, 'translation.npy'))
        self.rotation_matrix = np.load(os.path.join(Localizer.SETTINGS_PATH, 'rotation.npy'))

    def process_elements(self):
        while not self.stop_event.is_set():
            # Try to get element from queue
            #print('Trying to get element')
            element = None
            try:
                self.queue_access_mutex.acquire()
                pointcloud_np = self.queue.get(timeout=1.0)
                self.queue_access_mutex.release()
            except:
                self.queue_access_mutex.release()
                continue
            # Create RGB image from pointcloud
            rgb_image = []
            for i in range(0, pointcloud_np.shape[0]):
                rgb_image.append(float_to_rgb(pointcloud_np[i, 3]))
            rgb_image_np = np.asarray(rgb_image, dtype=np.uint8)
            rgb_image_np = np.reshape(rgb_image_np, (480, 640, 3))

            # Try to recognize the cup in image
            try:
                cup_bbox_list = self.cuprecognizer.predict_bboxes(rgb_image_np, pointcloud_np, self.translation_vector, self.rotation_matrix)
            except RuntimeError as e:
                print(e)
                cup_bbox_list = []


            # Try to inform observers
            try:
                if len(cup_bbox_list) != 0:
                    for observer in self.observers:
                        observer.localization_update(cup_bbox_list)
            except Exception as e:
                print('Exception while informing observers.')
                print(e)
            #bbox_list = result[0]
            #segmentation_list = result[1]

            # Recognize turtlebot
            try:
                # turtlebot_bbox = self.turtlebotrecognizer.predict_bounding_box(rgb_image_np, pointcloud_np, self.translation_vector, self.rotation_matrix, visualize=False)
                turtlebot_bbox = BBox()
		turtlebot_bbox.object_class = "turtlebot"
		turtlebot_bbox.x = -0.82
		turtlebot_bbox.y = 0.05
		turtlebot_bbox_list = []
                if turtlebot_bbox is not None:
                    turtlebot_bbox_list.append(turtlebot_bbox)
            except RuntimeError as e:
                print(e)
                turtlebot_bbox_list = []

            # Try to inform observers
            try:
                if len(turtlebot_bbox_list) != 0:
                    for observer in self.observers:
                        observer.localization_update(turtlebot_bbox_list)
            except Exception as e:
                print('Exception while informing observers.')
                print(e)

    def start_localizer(self):
        if self.initialized and not self.stop_event.is_set():
            return
        self.stop_event = threading.Event()
        self.queue_access_mutex = threading.Lock()
        self.thread = threading.Thread(target=self.process_elements)
        self.thread.daemon = True
        self.thread.start()


    def stop_localizer(self):
        if self.initialized and self.stop_event.is_set():
            return
        self.stop_event.set()



    # Scene should be a numpy array containing pointcloud data
    def localize_objects(self, pointcloud):
        if Localizer.LOG_INPUT:
            if os.path.isdir(Localizer.DATA_LOG_PATH):
                ts = calendar.timegm(time.gmtime())
                fname = str(ts) + '.npy'
                np.save(Localizer.DATA_LOG_PATH + '/' + fname, pointcloud)
                print('Logged frame')
            else:
                print('Data logging directory does not exist. Continuing.')
        self.queue_access_mutex.acquire()
        try:
            self.queue.put(pointcloud, block=False)
        except:
            self.queue.get()
            self.queue.put(pointcloud)
            #print('Queue is full. Element was replaced.')
        self.queue_access_mutex.release()

    def register_observer(self, observer):
        self.observers.append(observer)

    def unregister_observer(self, observer):
        self.observers.remove(observer)

if __name__=='__main__':
    print('Running localizer')
    relative_pointclouds_path = '../data/pointclouds5'
    pointclouds_path = os.path.join(os.path.dirname(__file__), relative_pointclouds_path)
    import glob
    files = glob.glob(os.path.join(pointclouds_path, '*.npy'))

    localizer = Localizer()
    localizer.initialize_localizer()
    localizer.start_localizer()
    max_count = 100
    for i, file in enumerate(files):
        print('Processing {}'.format(os.path.basename(file)))
        pointcloud_np = np.load(file)
        localizer.localize_objects(pointcloud_np)
        time.sleep(8.0)
        if i >= max_count -1:
            break
    localizer.stop_localizer()
    #visual = pcl.pcl_visualization.CloudViewing()
    #visual.ShowColorCloud(pointcloud, b'cloud')
    #time.sleep(30)


    #localizer.localize_objects(pointcloud_np)

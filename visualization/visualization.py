import os
import sys

import glob

import numpy as np

import skimage.io as io

import cv2

import matplotlib.pyplot as plt

from subprocess import call

rel_utils_dir = '../mrlocalization/utils'
utils_dir = os.path.join(os.path.dirname(__file__), rel_utils_dir)
sys.path.append(utils_dir)

from pcl_helper import float_to_rgb


rel_video_dir = '../data/video2'
video_dir = os.path.join(os.path.dirname(__file__), rel_video_dir)
sys.path.append(video_dir)

rel_out_dir = '../data/video_export'
out_dir = os.path.join(os.path.dirname(__file__), rel_out_dir)
sys.path.append(out_dir)

rel_svm_rec_dir = '../mrlocalization/cuprecognition/svm_recognizer'
svm_rec_dir =  os.path.join(os.path.dirname(__file__), rel_svm_rec_dir)
sys.path.append(svm_rec_dir)

rel_cup_rec_dir = '../mrlocalization/cuprecognition'
cup_rec_dir =  os.path.join(os.path.dirname(__file__), rel_cup_rec_dir)
sys.path.append(cup_rec_dir)

from svm_recognizer import SVMRecognizer
recognizer = SVMRecognizer()
recognizer.initialize_recognizer()

from cuprecognizer import CupRecognizer
cuprecognizer = CupRecognizer()
cuprecognizer.initialize_recognizer()


def create_video():
    # Load settings
    relative_settings_path = '../settings'
    settings_path = os.path.join(os.path.dirname(__file__),relative_settings_path)
    translation_vector = np.load(os.path.join(settings_path, 'translation.npy'))
    rotation_matrix = np.load(os.path.join(settings_path, 'rotation.npy'))

    # Get files
    files = glob.glob(video_dir + '/*.npy')

    # Render images
    for i, file in enumerate(files):
        print('Processing file {}'.format(os.path.basename(file)))
        pointcloud_np = np.load(file)
        rgb_image = get_rgb_image(pointcloud_np)

        recognizer_image = generate_recognizer_image(rgb_image, pointcloud_np, translation_vector, rotation_matrix)

        io.imsave(out_dir + '/{:03d}.png'.format(i), recognizer_image)

    # Create ideo with ffmpeg
    call('ffmpeg -r 1 -f image2 -s 640x480 -i {}/%03d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p -r 15 {}/test.mp4'.format(out_dir, out_dir).split())


def generate_recognizer_image(image, pointcloud_np, translation_vector, rotation_matrix):
    try:
        bboxes = recognizer.recognize(image, pointcloud_np)
        bboxes = []
    except:
        bboxes = []
    bbox_image = np.copy(image)
    for bbox in bboxes:
        cv2.rectangle(bbox_image,(bbox[1], bbox[0]),(bbox[1] + bbox[3], bbox[0] + bbox[2]),(0,255,0),3)
    try:
        cuprecognizer.predict_bboxes(image, pointcloud_np, translation_vector, rotation_matrix)
        bboxes2 = cuprecognizer.last_filtered_bboxes
    except RuntimeError as e:
        print(e)
        bboxes2 = []
    for bbox in bboxes2:
        cv2.rectangle(bbox_image,(bbox[1], bbox[0]),(bbox[1] + bbox[3], bbox[0] + bbox[2]),(255,0,0),3)
    return bbox_image



def get_rgb_image(pointcloud_np):
    rgb_image = []
    for i in range(0, pointcloud_np.shape[0]):
        rgb_image.append(float_to_rgb(pointcloud_np[i, 3]))
    rgb_image_np = np.asarray(rgb_image, dtype=np.uint8)
    rgb_image_np = np.reshape(rgb_image_np, (480, 640, 3))
    return rgb_image_np


if __name__ == '__main__':
    print('Running visualization script')
    create_video()

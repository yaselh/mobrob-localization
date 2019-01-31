# File requires working python.pcl
import os
import sys
import glob

from math import isclose

import numpy as np

import skimage.io as io
from skimage.viewer import ImageViewer
from skimage.viewer.canvastools import RectangleTool

import matplotlib.pyplot as plt

relative_utils_path = '../../../utils'
utils_path = os.path.join(os.path.dirname(__file__), relative_utils_path)
sys.path.append(utils_path)
from pcl_helper import float_to_rgb

import pcl

from svm_predictor import SVMPredictor

relative_dataset_path = '../../../../data/pointclouds7'
dataset_path = os.path.join(os.path.dirname(__file__), relative_dataset_path)

relative_output_dir = '../../../../data/pointclouds7/patches'
output_dir = os.path.join(os.path.dirname(__file__), relative_output_dir)


def get_rgb_data_from_pointcloud():
    pass

def get_rgb_image_from_pcd(path):
    pointcloud = pcl.load_XYZRGB(path)
    pointcloud_np = pointcloud.to_array()

    # Compute RGB image
    rgb_image = []
    for i in range(0, pointcloud_np.shape[0]):
        rgb_image.append(float_to_rgb(pointcloud_np[i, 3]))
    rgb_image_np = np.asarray(rgb_image, dtype=np.uint8)
    rgb_image_np = np.reshape(rgb_image_np, (480, 640, 3))
    return rgb_image_np

def get_rgb_image_from_npy(path):
    pointcloud_np = np.load(path)
    # Compute RGB image
    rgb_image = []
    for i in range(0, pointcloud_np.shape[0]):
        rgb_image.append(float_to_rgb(pointcloud_np[i, 3]))
    rgb_image_np = np.asarray(rgb_image, dtype=np.uint8)
    rgb_image_np = np.reshape(rgb_image_np, (480, 640, 3))
    return rgb_image_np

def annotate_data(annotation_file_path):
    print('No annotation file found. Annotation required.')
    files = glob.glob(os.path.join(dataset_path, '*.*'))
    annotations = []
    for file in files:
        print(file)
        file_extension = os.path.splitext(file)[1]
        if file_extension == '.pcd':
            image = get_rgb_image_from_pcd(file)
        elif file_extension == '.npy':
            image = get_rgb_image_from_npy(file)
        else:
            print('Unsupported file type.')
            continue
        viewer = ImageViewer(image)
        def enter_callback(extents):
            bbox = (int(extents[2]), int(extents[0]), int(extents[3] - extents[2]), int(extents[1] - extents[0]))
            if bbox != (0,0,1,0):
                annotations.append((file, bbox))
                print(bbox)
            else:
                print('No annotations were made.')
            viewer.close()
        rect_tool = RectangleTool(viewer, on_enter=enter_callback)
        viewer.show()
    annotation_str = ''
    for annotation in annotations:
        file_name = os.path.basename(annotation[0])
        bbox_row_min = annotation[1][0]
        bbox_col_min = annotation[1][1]
        bbox_rows = annotation[1][2]
        bbox_cols = annotation[1][3]
        with open(annotation_file_path, 'a') as text_file:
            print('{};{};{};{};{}'.format(file_name, bbox_row_min, bbox_col_min, bbox_rows, bbox_cols), file=text_file)

# Function Code taken from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """

    bb1_x1 = bb1[1]
    bb1_x2 = bb1[1] + bb1[3]
    bb1_y1 = bb1[0]
    bb1_y2 = bb1[0] + bb1[2]

    bb2_x1 = bb2[1]
    bb2_x2 = bb2[1] + bb2[3]
    bb2_y1 = bb2[0]
    bb2_y2 = bb2[0] + bb2[2]


    # determine the coordinates of the intersection rectangle
    x_left = max(bb1_x1, bb2_x1)
    y_top = max(bb1_y1, bb2_y1)
    x_right = min(bb1_x2, bb2_x2)
    y_bottom = min(bb1_y2, bb2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1_x2 - bb1_x1) * (bb1_y2 - bb1_y1)
    bb2_area = (bb2_x2 - bb2_x1) * (bb2_y2 - bb2_y1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def centroid_lies_in_bbox(centroid, bbox):
    centroid_within_row_range = centroid[0] >= bbox[0] and centroid[0] <= (bbox[0] + bbox[2])
    centroid_within_col_range = centroid[1] >= bbox[1] and centroid[1] <= (bbox[1] + bbox[3])
    return centroid_within_row_range and centroid_within_col_range

def extract_patches_from_image(image, height, width, stride, bbox):
    min_required_overlap = 0.4
    print('Processing image...')
    row_count = image.shape[0]
    col_count = image.shape[1]

    start_row = 0
    start_col = 0
    end_row = row_count
    end_col = col_count

    patches = [] # Format (row, col, height, width)
    for col in range(start_col, (end_col - width) + 1, stride):
        for row in range(start_row, (end_row - height) + 1, stride):
            patch = image[row:row+height, col:col+width]
            patch_bbox = (row, col, height, width)
            patch_bbox_centroid = (int(patch_bbox[0] + 0.5 * patch_bbox[2]), int(patch_bbox[1] + 0.5 * patch_bbox[3]))
            overlap = get_iou(bbox, patch_bbox)
            if overlap > min_required_overlap and centroid_lies_in_bbox(patch_bbox_centroid, bbox):
                #plt.imshow(patch)
                #plt.show()
                label = 1
            else:
                label = 0

            patches.append((patch, label))
    return patches

def extract_patches(annotation_file_path, percentage_positive):
    print('Extracting patches according to annotation file {}.'.format(annotation_file_path))
    with open(annotation_file_path, 'r') as f:
        data= f.read()
    annotation_strs = data.split('\n')
    overall_patches = []
    for annotation_str in annotation_strs:
        if not annotation_str:
            continue
        annotation_split = annotation_str.split(';')
        file_name = annotation_split[0]
        bbox_row_min = int(annotation_split[1])
        bbox_col_min = int(annotation_split[2])
        bbox_rows = int(annotation_split[3])
        bbox_cols = int(annotation_split[4])
        bbox = (bbox_row_min, bbox_col_min, bbox_rows, bbox_cols)

        file_extension = os.path.splitext(file_name)[1]
        if file_extension == '.pcd':
            image = get_rgb_image_from_pcd(os.path.join(dataset_path, file_name))
            patches = extract_patches_from_image(image, 70, 70, 4, bbox)
            overall_patches.extend(patches)
        elif file_extension == '.npy':
            image = get_rgb_image_from_npy(os.path.join(dataset_path, file_name))
            patches = extract_patches_from_image(image, 70, 70, 4, bbox)
            overall_patches.extend(patches)
        else:
            print('Unsupported file type.')
            continue
    # Select patches according to split ratio
    patches_np = np.array(overall_patches)
    np.random.shuffle(patches_np)

    labels_np = patches_np[:,1]
    positive_samples_mask = labels_np == 1
    negative_samples_mask = labels_np == 0
    positive_count = positive_samples_mask.sum()

    if isclose(percentage_positive, 0.0):
        negative_count = negative_samples_mask.sum()
    else:
        percentage_negative = 1.0 - percentage_positive
        one_percent = positive_count / int(percentage_positive * 100)

        negative_count = int(percentage_negative * 100 *one_percent)

    positive_samples = patches_np[positive_samples_mask]
    negative_samples = patches_np[negative_samples_mask]
    selected_patches = []
    for i in range(0, positive_count):
        selected_patches.append(positive_samples[i])
    for i in range(0, negative_count):
        selected_patches.append(negative_samples[i])
    return np.array(selected_patches)

def write_patches_to_dir(patches, output_dir):
    print('Writing patches to filesystem...')
    for i, patch in enumerate(patches):
        if i % 100 == 0:
            print('File {}/{}'.format(i, len(patches)))
        image = patch[0]
        label = patch[1]
        if label == 1:
            io.imsave(os.path.join(output_dir, 'pos', str(i) + '.png'), image)
        else:
            io.imsave(os.path.join(output_dir, 'neg', str(i) + '.png'), image)




def create_dataset():
    annotation_file_path = os.path.join(dataset_path, 'cup_annotations.txt')
    if not os.path.isfile(annotation_file_path):
        annotate_data(annotation_file_path)
    patches = extract_patches(annotation_file_path, 0.2)
    write_patches_to_dir(patches, output_dir)

def perform_hn_mining():
    annotation_file_path = os.path.join(dataset_path, 'cup_annotations.txt')
    if not os.path.isfile(annotation_file_path):
        annotate_data(annotation_file_path)
    patches = extract_patches(annotation_file_path, 0.0)
    svm_predictor = SVMPredictor()
    svm_predictor.initialize_predictor()
    sample_count = min(100000, patches.shape[0])
    samples = patches[0:sample_count-1, 0]
    print('Predicting samples. This can take a while.')
    results = svm_predictor.predict(samples)
    false_positives = []
    for result, patch in zip(results, patches):
        if result[1] > 0.5 and patch[1] == 0:
            false_positives.append(patch)
    write_patches_to_dir(false_positives, output_dir)




if __name__=='__main__':
    #create_dataset()
    perform_hn_mining()

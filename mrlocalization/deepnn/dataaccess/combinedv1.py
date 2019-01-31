import os
import glob
import keras
from keras.preprocessing.image import img_to_array, load_img
import zipfile
import numpy as np
import sys
from skimage.transform import resize
from skimage.exposure import rescale_intensity


dataset_dir = '../../../data/processed/combinedv1'
download_url_val = 'http://cloud.dk-s.de/datasets/combinedv1/combinedv1_val.zip'

def parse_data():
    pass

def load_data():
    # Check if dataset directory already exists
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

def load_training_data():
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    target_path = os.path.abspath(dataset_dir + '/combinedv1_train.zip')
    if not os.path.isfile(target_path):
        keras.utils.get_file(fname=target_path, origin=download_url_val)
        print('Unzipping files...')
        zip_ref = zipfile.ZipFile(target_path, 'r')
        zip_ref.extractall(os.path.dirname(target_path))
        zip_ref.close()


def load_validation_data():
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    target_path = os.path.abspath(dataset_dir + '/combinedv1_val.zip')
    if not os.path.isfile(target_path):
        keras.utils.get_file(fname=target_path, origin=download_url_val)
        print('Unzipping files...')
        zip_ref = zipfile.ZipFile(target_path, 'r')
        zip_ref.extractall(os.path.dirname(target_path))
        zip_ref.close()
    # Load input files
    input_files_list = glob.glob(dataset_dir + '/val/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].jpg')
    print len(input_files_list)
    sys.exit()
    x_val = np.empty((len(input_files_list), 224, 224, 3))
    for i in range(0, len(input_files_list)):
        path = input_files_list[i]
        image = load_img(path)
        x_element = img_to_array(image)
        x_element = rescale_intensity(x_element)
        x_element = resize(x_element, (224, 224))
        x_val[i] = x_element
    return x_val


if __name__ == '__main__':
    load_validation_data()

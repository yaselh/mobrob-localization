import os
import glob

import numpy as np

import skimage
from skimage.filters import gaussian
from skimage.feature import hog
from skimage.transform import resize
from skimage.color import rgb2gray
import skimage.io as io

from sklearn import svm, metrics
from sklearn.externals import joblib

import time

import matplotlib.pyplot as plt

relative_dataset_path_train = '../../../../data/processed/own_datav7/train'
dataset_path_train = os.path.join(os.path.dirname(__file__), relative_dataset_path_train)

relative_dataset_path_val = '../../../../data/processed/own_datav7/val'
dataset_path_val = os.path.join(os.path.dirname(__file__), relative_dataset_path_val)


def extract_features(image):
    # If rgb image convert to grayscale
    if len(image.shape) > 2:
        image = rgb2gray(image)
    image = resize(image, (32,32), mode='constant') # Important to realize that here range is converted to 0.0-1.0
    image = gaussian(image, sigma=1.0)
    # plt.imshow(image)
    # plt.show()
    #image = pad_image(image,64)
    # Original setting for old classifier: orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1). Also some used block_norm L1
    fd = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=False, block_norm='L2')
    return fd

def load_data(path):
    pos_paths = [path + '/pos/*.png']
    neg_paths = [path + '/neg/*.png', path + '/hn1/*.png', path + '/hn2/*.png', path + '/hn3/*.png']
    pos_files = []
    neg_files = []
    for path in pos_paths:
        cur_pos_files = glob.glob(path)
        pos_files.extend(cur_pos_files)
    for path in neg_paths:
        cur_neg_files = glob.glob(path)
        neg_files.extend(cur_neg_files)

    pos_features = []
    for i, file in enumerate(pos_files):
        image = io.imread(file, as_grey=True)
        fd = extract_features(image)
        pos_features.append(fd)
        print(i)

    neg_features = []
    for i, file in enumerate(neg_files):
        image = io.imread(file, as_grey=True)
        fd = extract_features(image)
        neg_features.append(fd)
        print(i)
    return pos_features, neg_features

def train_svm_binary(pos_features, neg_features):
    all_features_list = []
    all_features_list.extend(pos_features)
    all_features_list.extend(neg_features)
    pos_class_list = [1] * len(pos_features)
    neg_class_list = [0] * len(neg_features)
    all_classes_list = []
    all_classes_list.extend(pos_class_list)
    all_classes_list.extend(neg_class_list)

    print('Training SVM...')
    classifier = svm.SVC(C=32.0, probability=True)
    classifier.fit(all_features_list, all_classes_list)
    return classifier

def evaluate_svm_binary(pos_features, neg_features, classifier):
    all_features_list = []
    all_features_list.extend(pos_features)
    all_features_list.extend(neg_features)
    pos_class_list = [1] * len(pos_features)
    neg_class_list = [0] * len(neg_features)
    all_classes_list = []
    all_classes_list.extend(pos_class_list)
    all_classes_list.extend(neg_class_list)

    start_time = time.time()
    predicted = classifier.predict(all_features_list)
    probs = classifier.predict_proba(all_features_list)
    print(probs)
    end_time = time.time()
    print('Classification took {} seconds.'.format(end_time-start_time))

    print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(all_classes_list, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(all_classes_list, predicted))
    print("\nAccuracy: %s" % metrics.accuracy_score(all_classes_list ,predicted))


    #probs = classifier.predict_proba(all_features_list)
    #print(probs)

if __name__=='__main__':
    val_active = True
    if val_active:
        pos_features_val, neg_features_val = load_data(dataset_path_val)
        classifier = joblib.load('test_svm.pkl')
        evaluate_svm_binary(pos_features_val, neg_features_val, classifier)
    else:
        pos_features_train, neg_features_train = load_data(dataset_path_train)
        classifier = train_svm_binary(pos_features_train, neg_features_train)
        joblib.dump(classifier, 'test_svm.pkl')

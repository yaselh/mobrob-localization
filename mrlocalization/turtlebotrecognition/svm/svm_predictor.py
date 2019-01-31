import os
import skimage.io as io
import numpy as np
import glob
from sklearn.externals import joblib
from svm_train import extract_features
import matplotlib.pyplot as plt

class SVMPredictor:

    model_path = os.path.join(os.path.dirname(__file__), 'test_svm.pkl')

    def __init__(self):
        self.classifier = joblib.load(SVMPredictor.model_path)


    def predict(self, images):
        feature_descriptors = []
        for image in images:
            fd = extract_features(image)
            feature_descriptors.append(fd)
        predicted = self.classifier.predict_proba(feature_descriptors)
        return predicted


if __name__=='__main__':

    img_path = "/home/yassinel/frame0000.jpg"
    img = io.imread(img_path, as_grey=True)

    patches = image.extract_patches_2d(img, (100, 100), max_patches=50)
    print(patches.shape)
    threshold = 0.90
    predictor = SVMPredictor()
    for patch, predictions in zip(patches, predictor.predict(patches)):
        if np.argmax(predictions) == 1 and predictions[1] >= threshold:
            plt.imshow(patch)
            plt.show()

import os
from keras.models import load_model
import keras
import tensorflow as tf
import numpy as np
import skimage
import skimage.io as io
import matplotlib.pyplot as plt

class SWNetPredictor:

    relative_model_path = '../../../../models/swnetv4/model_epoch_95.hdf5'
    model_path = os.path.join(os.path.dirname(__file__), relative_model_path)

    def initialize_predictor(self):
        keras.backend.clear_session()
        self.model = load_model(self.model_path)
        # The following line is necessary as otherwise prediction will fail as it is run
        # in another thread. See https://github.com/keras-team/keras/issues/2397
        self.graph = tf.get_default_graph()


    def predict(self, images):
        for i in range(0, len(images)):
            images[i] = skimage.img_as_float(images[i])
            if images[i].shape != (50,50, 3):
                images[i] = skimage.transform.resize(images[i], [50,50,3])
        batch_size = len(images)
        images_np = np.asarray(images)
        # The following line is necessary as otherwise prediction will fail as it is run
        # in another thread. See https://github.com/keras-team/keras/issues/2397
        with self.graph.as_default():
            result = self.model.predict(images_np, batch_size=batch_size)
        return result[:,0]



if __name__ == '__main__':
    predictor = SPNetPredictor()
    predictor.initialize_predictor()
    #relative_image_path = 'C:/Users/Daniel/Code/datasets/thur15k/CoffeeMug/Src/60.jpg'
    relative_image_path = '../../../../data/processed/cupv2/train_images/000000023451.jpg'
    image_path = os.path.join(os.path.dirname(__file__), relative_image_path)
    image = io.imread(image_path)
    res = predictor.predict(image)

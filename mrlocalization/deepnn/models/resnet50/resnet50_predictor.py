import resnet50_model
import os
import numpy as np
from keras.applications.imagenet_utils import preprocess_input

class Resnet50_Predictor:

    relative_model_path = '../../../../models/resnet50/combinedv8/model_epoch_1.hdf5'
    model_path = os.path.join(os.path.dirname(__file__), relative_model_path)

    def initialize_predictor(self):
        self.model = resnet50_model.load_model_from_file_or_url(self.model_path)
        self.stride = resnet50_model.get_stride()

    # image must be (Height, Width, 3) where 3 is for R, G, B
    def predict(self, image):
        # crop if image dims do not match stride
        stride = resnet50_model.get_stride()
        w_rest = image.shape[0] % self.stride
        h_rest = image.shape[1] % self.stride

        if w_rest > 0:
            w_crop_1 = int(np.round(w_rest / 2))
            w_crop_2 = int(w_rest - w_crop_1)
            image = image[w_crop_1:-w_crop_2,:,:]

        if h_rest > 0:
            h_crop_1 = int(np.round(h_rest / 2))
            h_crop_2 = int(h_rest - h_crop_1)
            image = image[:,h_crop_1:-h_crop_2,:]

        image = np.array(image,dtype='float')
        image = image[np.newaxis,...]

        image = preprocess_input(image)

        prediction_result = self.model.predict(image, batch_size = 1)
        return prediction_result[0]

if __name__ == '__main__':
    predictor = Resnet50_Predictor()
    predictor.initialize_predictor()
    import skimage.io as io
    relative_image_path = '../../../../data/processed/combinedv7/val_images/000000002592.jpg'
    image_path = os.path.join(os.path.dirname(__file__), relative_image_path)
    image = io.imread(image_path)
    prediction = predictor.predict(image)
    max_map = np.argmax(prediction, axis=2)
    cup_prediction = max_map == 1
    import matplotlib.pyplot as plt
    plt.imshow(cup_prediction)
    plt.show()

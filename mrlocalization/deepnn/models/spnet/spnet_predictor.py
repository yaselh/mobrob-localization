import os
from keras.models import load_model
from image_preprocessing import resize_image, segment_image
from data_selector import generate_patches
import numpy as np
import skimage
import skimage.io as io
import matplotlib.pyplot as plt

class SPNetPredictor:

    relative_model_path = '../../../../models/spnet/simple_thur_patches_v1/model_epoch_36_sgd.hdf5'
    model_path = os.path.join(os.path.dirname(__file__), relative_model_path)

    def initialize_predictor(self):
        self.model = load_model(self.model_path)

    def predict(self, image):
        image = resize_image(image, 200, 200)
        segments = segment_image(image, n_segments = 200, compactness=10, sigma=1)
        patches = generate_patches(image, segments, context_window_ratio = 2, square_data_size=150, min_patch_size=50)
        for i in range(0, len(patches)):
            patches[i] = skimage.img_as_float(patches[i]) # Check if this conversion works for adjusting range!!!!
        num_patches = len(patches)
        patches_np = np.asarray(patches)
        result = self.model.predict(patches_np, batch_size=num_patches)
        # get top 20 percent
        result_sorted = np.sort(result, axis=0)
        best_x_percent = 100
        twenty_percent_of_elements = int((result_sorted.shape[0] / 100) * best_x_percent)
        threshold = result_sorted[result_sorted.shape[0]-twenty_percent_of_elements, 0]


        result_image = np.zeros_like(segments, dtype=np.float32)
        for i in range(0, result.shape[0]):
            mask = segments == i
            if result[i,0] >= threshold and result[i,0] > 0.5:
                result_image[mask == True] = result[i,0]
            else:
                result_image[mask == True] = 0.0
        plt.imshow(result_image)
        plt.show()


if __name__ == '__main__':
    predictor = SPNetPredictor()
    predictor.initialize_predictor()
    #relative_image_path = 'C:/Users/Daniel/Code/datasets/thur15k/CoffeeMug/Src/60.jpg'
    relative_image_path = '../../../../data/processed/cupv2/train_images/000000000605.jpg'
    image_path = os.path.join(os.path.dirname(__file__), relative_image_path)
    image = io.imread(image_path)
    res = predictor.predict(image)

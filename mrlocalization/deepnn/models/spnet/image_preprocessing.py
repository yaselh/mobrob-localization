import skimage
import skimage.io as io
from skimage.segmentation import  slic, felzenszwalb
from skimage.segmentation import mark_boundaries
import time
import matplotlib.pyplot as plt
import numpy as np




def segment_image(image, n_segments=200, compactness=10, sigma=1):
    segments = slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma, convert2lab=True)
    #segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    return segments

def resize_image(image, max_height=200, max_width=200):
    height = image.shape[0]
    width = image.shape[1]
    ratio = min(max_height / height, max_width / width)
    target_height = round(height * ratio)
    target_width = round(width * ratio)
    image = skimage.img_as_ubyte(skimage.transform.resize(image, (target_height, target_width)))
    return image



if __name__=='__main__':
    image_file_path = '../../../../data/processed/cupv2/train_images/000000042109.jpg'
    label_file_path = '../../../../data/processed/cupv2/train_labels/000000042109.png'
    image = io.imread(image_file_path)
    label_image = io.imread(label_file_path, as_grey=True)
    image = resize_image(image, 200, 200)
    label_image = resize_image(label_image, 200, 200)
    segments = segment_image(image, n_segments = 200, compactness=10, sigma=1)
    plt.imshow(mark_boundaries(image,segments))
    plt.show()
    #patches, labels = generate_patches_with_labels(image, label_image, segments, context_window_ratio = 1)
    #print(len(patches))
    #print(len(labels))

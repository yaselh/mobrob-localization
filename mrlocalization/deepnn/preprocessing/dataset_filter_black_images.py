# Script filters out completely black images from label directory and deletes corresponding
# image files
import glob
import skimage
import skimage.io as io
import numpy as np
from os.path import basename
import os

classes_to_clear = [3]

label_dir = 'C:/Users/Daniel/Code/python/mobrob-localization/data/processed/combinedv8/val_labels'
image_dir = 'C:/Users/Daniel/Code/python/mobrob-localization/data/processed/combinedv8/val_images'

mask_files =  glob.glob(label_dir + '/*.png')
for i, file in enumerate(mask_files):
    image = skimage.img_as_ubyte(io.imread(file, as_grey=True))
    if np.count_nonzero(image) == 0:
        print('Removing file: ' + basename(file))
        os.remove(file)
        os.remove(image_dir + '/' + basename(file)[:-3] + 'jpg')

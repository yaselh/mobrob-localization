# Lists images in directory. For each image sets value of pixels that match
# one of the combined classes to 0
import glob
import skimage
import skimage.io as io
import numpy as np
from os.path import basename

classes_to_clear = [2]

dataset_dir = 'C:/Users/Daniel/Code/python/mobrob-localization/data/processed/combinedv7/val_labels'
output_dir = 'C:/Users/Daniel/Code/python/mobrob-localization/data/processed/combinedv8/val_labels'

mask_files =  glob.glob(dataset_dir + '/*.png')
for i, file in enumerate(mask_files):
    image = skimage.img_as_ubyte(io.imread(file, as_grey=True))
    for class_value in classes_to_clear:
        image[image==classes_to_clear] = 0
    print('Saving picture {} of {}.'.format(i, len(mask_files)))
    io.imsave(output_dir + '/' + basename(mask_files[i]), image)

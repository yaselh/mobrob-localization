import glob
import numpy as np
import skimage.io as io
import skimage
from os.path import basename

old_labels = [100, 150, 200]
new_labels = [1, 2, 3]

dataset_dir = 'D:/mobrob_datasets/combinedv5/train_labels'
output_dir = 'D:/mobrob_datasets/combinedv6/train_labels'

mask_files =  glob.glob(dataset_dir + '/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].png')

for i in range(0, len(mask_files)):
    image = skimage.img_as_ubyte(io.imread(mask_files[i], as_grey=True))
    for old_label, new_label in zip(old_labels, new_labels):
        image[image==old_label] = new_label

    print('Saving picture {} of {}.'.format(i, len(mask_files)))
    io.imsave(output_dir + '/' + basename(mask_files[i]), image)

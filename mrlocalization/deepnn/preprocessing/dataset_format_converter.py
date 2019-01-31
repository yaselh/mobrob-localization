import skimage.io as io
import skimage
import glob
import numpy as np
from os.path import basename


dataset_dir = 'D:/mobrob_datasets/cupv1/train_labels'
output_dir = 'D:/mobrob_datasets/cupv2/train_labels'

mask_files =  glob.glob(dataset_dir + '/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9].png')

# Define how many classes exist in generated dataset
class_count = 1
# Define which value should be assigned to the classes in the combined image
class_values = [1]

if len(mask_files) % class_count != 0:
    print('Error: No valid filed count!')
    exit(0)
for i in range(0, len(mask_files), class_count):
    combined_labels = None
    for j, value in enumerate(class_values):
        image = skimage.img_as_ubyte(io.imread(mask_files[i + j], as_grey=True))
        image = np.clip(image, 0, class_values[j])
        if combined_labels is None:
            combined_labels = image
        else:
            mask = image > 0
            combined_labels = np.where(mask, image, combined_labels)
    print('Saving picture {} of {}.'.format(int(i/class_count), int(len(mask_files)/class_count)))
    io.imsave(output_dir + '/' + basename(mask_files[i])[:-6] + '.png', combined_labels)

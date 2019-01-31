# Script that generates easy to process dataset with the classes cup, table and unknown from mscoco dataset
import sys
mscoco_python_path = '../../../3rdparty/cocoapi/PythonAPI'
sys.path.append(mscoco_python_path)

from random import randint
from shutil import copyfile
import skimage
import skimage.io as io
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

# Specify input files directory path, annotation file path and output files directory here.
# This script can be used for generating train, val and test data.
data_dir = 'D:/mobrob_datasets/mscoco/train2017'
ann_file = 'D:/mobrob_datasets/mscoco/annotations_trainval2017/instances_train2017.json'
output_dir_masks = 'D:/mobrob_datasets/combinedv4/train_labels'
output_dir_images = 'D:/mobrob_datasets/combinedv4/train_images'
coco=COCO(ann_file)

# Display all categories
#print('Dataset contains the following categories:')
#categories = coco.loadCats(coco.getCatIds())
#for category in categories:
#    print(category['name'])

# Load all images containing cups
cup_category_id = coco.getCatIds(catNms=['cup']);
cup_image_ids = coco.getImgIds(catIds=cup_category_id);
print('Training data contains {0} images that contain cups.'.format(len(cup_image_ids)))

# Load all images containing tables
table_category_id = coco.getCatIds(catNms=['dining table']);
table_image_ids = coco.getImgIds(catIds=table_category_id);
print('Training data contains {0} images that contain tables.'.format(len(table_image_ids)))

# Load all images for selection of "other"-category
other_category_ids = coco.getCatIds()
other_category_ids.remove(cup_category_id[0])
other_category_ids.remove(table_category_id[0])
other_image_ids = coco.getImgIds(); # load all image IDs here cause also images with cups should be selected
print('Training data contains {0} images overall.'.format(len(other_image_ids)))

#Here selection strategy for other pictures has to be implemented. Some that also contain cups and tables and some that don't?
cup_class_percentage = 1/3
table_class_percentage = 1/3
other_class_percentage = 1/3
class_count = 3

# Check which class contains the least images (assumes that other_image_ids always has enough other data so it is not considered here)
min_image_count = min(len(cup_image_ids), len(table_image_ids))
overall_count = class_count * min_image_count

# Creation of the dataset
def create_mask_outputs(image_id, output_classes, coco_categories):
    image = coco.loadImgs(image_id)[0]
    input_file_path = data_dir + '/' + image['file_name']
    output_file_path = output_dir_images + '/' + image['file_name']
    original_image = io.imread(input_file_path)
    image_shape = original_image.shape
    io.imsave(output_file_path, original_image)
    for class_id, categories in zip(output_classes, coco_categories):
        annotation_ids = coco.getAnnIds(imgIds=image['id'], catIds=categories, iscrowd=None)
        annotations = coco.loadAnns(annotation_ids)
        combined_rle = None
        for annotation in annotations:
            rle = coco.annToRLE(annotation)
            if combined_rle is not None:
                combined_rle = mask_utils.merge([combined_rle, rle], intersect = False) # -------------------wrong, continue coding here!
            else:
                combined_rle = rle
        mask_output_file_path = output_dir_masks + '/' + image['file_name']
        path_split = mask_output_file_path.split('.')
        del path_split[-1]
        mask_output_file_path = ''.join(path_split)
        mask_output_file_path = mask_output_file_path + '_' + str(class_id) + '.png'
        if combined_rle is not None:
            combined_mask = mask_utils.decode(combined_rle)
            combined_mask = skimage.img_as_ubyte(combined_mask)
            combined_mask = exposure.rescale_intensity(combined_mask)
            io.imsave(mask_output_file_path, combined_mask)
#            plt.imshow(combined_mask) # show combined mask image
#            plt.show()
        else:
            empty_image = np.zeros(shape=image_shape, dtype='ubyte', order='C')
            io.imsave(mask_output_file_path, empty_image)

selected_image_ids = set()
# Cup data selection
cup_count = int(cup_class_percentage * overall_count)
for i in range(0, cup_count):
    rand_index = randint(0, len(cup_image_ids) - 1)
    id = cup_image_ids.pop(rand_index)
    selected_image_ids.add(id)

# Table data selection
table_count = int(table_class_percentage * overall_count)
for i in range(0, table_count):
    rand_index = randint(0, len(table_image_ids) - 1)
    id = table_image_ids.pop(rand_index)
    selected_image_ids.add(id)

# Other data selection sampled randomly from all data except the data already used
other_count = int(other_class_percentage * ((len(selected_image_ids) / (class_count - 1)) * class_count))
print('After adding all class images there are {} images added. Adding {} images belonging to class \"other\".'.format(len(selected_image_ids), other_count))
other_image_ids_available = set(other_image_ids) - selected_image_ids
other_image_ids_available = list(other_image_ids_available)
if len(other_image_ids_available) < other_count:
    print('Warning: Not enough other data ({}) for balancing with other classes. Possible imbalance!'.format(len(other_image_ids_available)))
    input("Press Enter to continue...")
for i in range(0, other_count):
    rand_index = randint(0, len(other_image_ids_available) - 1)
    id = other_image_ids_available.pop(rand_index)
    selected_image_ids.add(id)


# write mask files
print("-------------------Generating Files---------------------")
overall_image_count = len(selected_image_ids)
for i, id in enumerate(selected_image_ids):
    print('Generating image ' + str(i) + '/' + str(overall_image_count) + '...')
    create_mask_outputs(id, [1, 2, 3], [cup_category_id, table_category_id, other_category_ids])

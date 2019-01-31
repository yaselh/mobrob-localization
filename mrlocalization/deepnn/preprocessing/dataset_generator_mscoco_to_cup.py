
import skimage
import skimage.io as io
from skimage import exposure
import numpy as np
import sys
mscoco_python_path = '../../../3rdparty/cocoapi/PythonAPI'
sys.path.append(mscoco_python_path)
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

data_dir = 'D:/mobrob_datasets/mscoco/val2017'
ann_file = 'D:/mobrob_datasets/mscoco/annotations_trainval2017/instances_val2017.json'
output_dir_masks = 'D:/mobrob_datasets/cupv1/val_labels'
output_dir_images = 'D:/mobrob_datasets/cupv1/val_images'
coco=COCO(ann_file)

#categories = coco.loadCats(coco.getCatIds())
#for category in categories:
#    print(category['name'])


cup_category_id = coco.getCatIds(catNms=['cup']);
cup_image_ids = coco.getImgIds(catIds=cup_category_id);
print('Training data contains {0} images that contain cups.'.format(len(cup_image_ids)))

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

for i, id in enumerate(cup_image_ids):
    print('Generating image ' + str(i) + '/' + str(len(cup_image_ids)) + '...')
    create_mask_outputs(id, [0], [cup_category_id])

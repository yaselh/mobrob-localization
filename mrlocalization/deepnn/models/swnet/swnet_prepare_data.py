import sys
mscoco_python_path = '../../../../3rdparty/cocoapi/PythonAPI'
sys.path.append(mscoco_python_path)

from pycocotools.coco import COCO
import skimage.io as io
import random

data_dir = 'D:/mobrob_datasets/mscoco/val2017'
ann_file = 'D:/mobrob_datasets/mscoco/annotations_trainval2017/instances_val2017.json'
output_dir_pos = 'D:/mobrob_datasets/cup_bboxv3/pos'
output_dir_neg = 'D:/mobrob_datasets/cup_bboxv3/neg'
coco=COCO(ann_file)

# Load all images containing cups
cup_category_id = coco.getCatIds(catNms=['cup']);
cup_image_ids = coco.getImgIds(catIds=cup_category_id);
print('Training data contains {0} images that contain cups.'.format(len(cup_image_ids)))


def save_instances_from_annotations(image, annotations, output_dir, min_width_height = 50, context_window_ratio = 1/3):
    saved_images = 0
    input_file_path = data_dir + '/' + image['file_name']
    img = io.imread(input_file_path)
    # Check if image is RGB
    if len(img.shape) < 3:
        return saved_images
    for i, annotation in enumerate(annotations):
        bbox = annotation['bbox']
        left_upper_col = bbox[0]
        left_upper_row = bbox[1]
        width = bbox[2]
        height = bbox[3]

        # Check if min size is met in width and height
        if width < min_width_height or height < min_width_height:
            continue

        # Recalculate bounding box from context window
        if context_window_ratio != 0:
            row_center = int(left_upper_row + (height / 2))
            col_center = int(left_upper_col + (width / 2))

            target_width = width + width * context_window_ratio
            target_height = height + height * context_window_ratio

            target_row_min = int(max(0, row_center - (target_height / 2)))
            target_row_max = int(min(img.shape[0] - 1, row_center + (target_height / 2)))
            target_col_min = int(max(0, col_center - (target_width / 2)))
            target_col_max = int(min(img.shape[1] - 1, col_center + (target_width / 2)))

            max_half_width = int(min(col_center - target_col_min, target_col_max - col_center))
            max_half_height =int(min(row_center - target_row_min, target_row_max - row_center))

            row_min = int(max(0, row_center - max_half_height))
            row_max = int(min(img.shape[0] - 1, row_center + max_half_height))
            col_min = int(max(0, col_center - max_half_width))
            col_max = int(min(img.shape[1] - 1, col_center + max_half_width))
        else:
            col_min = int(left_upper_col)
            col_max = int(left_upper_col + width)
            row_min = int(left_upper_row)
            row_max = int(left_upper_row + height)

        cropped_img = img[row_min:row_max, col_min:col_max, :]
        output_path = output_dir + '/{}_{}.png'.format(image['id'], i)
        io.imsave(output_path, cropped_img)
        saved_images = saved_images + 1
    return saved_images

goal_count = 0
for i, image_id in enumerate(cup_image_ids):
    print('Cropping cups for image {}/{}'.format(i+1, len(cup_image_ids)))
    image = coco.loadImgs(image_id)[0]
    annotation_ids = coco.getAnnIds(imgIds=image['id'], catIds=cup_category_id, iscrowd=None)
    annotations = coco.loadAnns(annotation_ids)
    saved_images = save_instances_from_annotations(image, annotations, output_dir_pos, min_width_height = 50, context_window_ratio = 2/3)
    goal_count = goal_count + saved_images
# ----------------------Select other data-------------------------

other_category_ids = coco.getCatIds()
other_category_ids.remove(cup_category_id[0])
other_image_ids = coco.getImgIds(); # load all image IDs here cause also images with cups should be selected
random.shuffle(other_image_ids)
print('Training data contains {0} images overall.'.format(len(other_image_ids)))

current_crop_count = 0
for i, image_id in enumerate(other_image_ids):
    print('Cropping objects for image {}'.format(i+1))
    image = coco.loadImgs(image_id)[0]
    annotation_ids = coco.getAnnIds(imgIds=image['id'], catIds=other_category_ids, iscrowd=None)
    annotations = coco.loadAnns(annotation_ids)
    saved_images = save_instances_from_annotations(image, annotations, output_dir_neg, min_width_height = 50, context_window_ratio = 2/3)
    current_crop_count = current_crop_count + saved_images
    if current_crop_count >= goal_count:
        break

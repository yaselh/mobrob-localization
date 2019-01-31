import numpy as np
import glob
from image_preprocessing import resize_image, segment_image
import os
import random
import skimage.io as io
import skimage


def pad_image(img,img_size) :
	max_dim = np.argmax(img.shape)
	min_dim = 1 - max_dim

	#resize the largest dim to img_size
	#if img.shape[max_dim] >= img_size:
	resize_factor = np.float(img_size) / np.float(img.shape[max_dim])
	new_min_dim_size = np.round( resize_factor * np.float(img.shape[min_dim]) )
	new_size = [img_size,img_size,3]
	new_size[min_dim] = new_min_dim_size

	img = skimage.img_as_ubyte(skimage.transform.resize(np.uint8(img), new_size, preserve_range=False))

	# pad dims
	pad_max = img_size - img.shape[max_dim]
	pad_min = img_size - img.shape[min_dim]

	pad = [[0,0],[0,0]]
	pad[max_dim][0] = np.int(np.round(pad_max / 2.0))
	pad[max_dim][1] = np.int(pad_max - pad[max_dim][0])

	pad[min_dim][0] = np.int(np.round(pad_min / 2.0))
	pad[min_dim][1] = np.int(pad_min - pad[min_dim][0])

	pad_tuple = ( (pad[0][0],pad[0][1]), (pad[1][0],pad[1][1]), (0,0))
	img = np.pad(img,pad_tuple,mode='constant')

	return img

def generate_patch_from_bbox(image, row_min, row_max, col_min, col_max, context_window_ratio, min_patch_size):
    sub_image_height = row_max - row_min
    sub_image_width = col_max - col_min

    # Increase context with factor context_window_ratio
    sub_image_target_height = sub_image_height + sub_image_height * context_window_ratio
    sub_image_target_width = sub_image_width + sub_image_width * context_window_ratio
    if min_patch_size != 0:
        sub_image_target_height = max(min_patch_size, sub_image_target_height)
        sub_image_target_width = max(min_patch_size, sub_image_target_width)

    row_center = int(row_max - (sub_image_height / 2))
    col_center = int(col_max - (sub_image_width / 2))

    target_row_min = int(max(0, row_center - (sub_image_target_height / 2)))
    target_row_max = int(min(image.shape[0] - 1, row_center + (sub_image_target_height / 2)))
    target_col_min = int(max(0, col_center - (sub_image_target_width / 2)))
    target_col_max = int(min(image.shape[1] - 1, col_center + (sub_image_target_width / 2)))

    # Make sure that previous center point stays centered. This would not be the Increase
    # if a patch lies on the boundary of an image
    resize_length_row = min(target_row_max - row_max, row_min - target_row_min)
    resize_length_col = min(target_col_max - col_max, col_min - target_col_min)

    row_min = row_min - resize_length_row
    row_max = row_max + resize_length_row
    col_min = col_min - resize_length_col
    col_max = col_max + resize_length_col

    sub_image = image[row_min:row_max, col_min:col_max, :]
    return sub_image

def generate_patches(image, segments, context_window_ratio = 1/3, square_data_size = 0, min_patch_size=0):
    num_patches = np.max(segments)
    patches = []
    for i in range(0, num_patches):
        current_segment = (segments == i)
        row_non_zero, col_non_zero = np.where(current_segment)
        col_min = np.min(col_non_zero)
        col_max = np.max(col_non_zero)
        row_min = np.min(row_non_zero)
        row_max = np.max(row_non_zero)
        # Calculate label
        row_middle = int(row_max-((row_max-row_min)/2))
        col_middle = int(col_max-((col_max-col_min)/2))
        sub_image = generate_patch_from_bbox(image, row_min, row_max, col_min, col_max, context_window_ratio, min_patch_size)

        # If square_data_size is not zero resize to square with padding
        if square_data_size != 0:
            sub_image = pad_image(sub_image, square_data_size)

        patches.append(sub_image)
    return patches

def generate_patches_with_labels(image, label_image, segments, context_window_ratio = 1/3, square_data_size = 0, min_patch_size=0):
    num_patches = np.max(segments)
    patches = []
    labels = []
    for i in range(0, num_patches):
        current_segment = (segments == i)
        row_non_zero, col_non_zero = np.where(current_segment)
        col_min = np.min(col_non_zero)
        col_max = np.max(col_non_zero)
        row_min = np.min(row_non_zero)
        row_max = np.max(row_non_zero)
        # Calculate label
        row_middle = int(row_max-((row_max-row_min)/2))
        col_middle = int(col_max-((col_max-col_min)/2))
        if np.max(label_image == 255):
            label_image[label_image == 255] = 1
        label_value = label_image[row_middle, col_middle]
        sub_image = generate_patch_from_bbox(image, row_min, row_max, col_min, col_max, context_window_ratio, min_patch_size)

        # If square_data_size is not zero resize to square with padding
        if square_data_size != 0:
            sub_image = pad_image(sub_image, square_data_size)

        #plt.imshow(sub_image)
        #plt.show()
        patches.append(sub_image)
        labels.append(label_value)
    return patches, labels


def select_patches_50_to_50(patches, labels, num_max_patches_per_image=10):
    num_positive_labels = labels.count(1)
    num_negative_labels = labels.count(0)

    minimum_num_labels = min(num_positive_labels, num_negative_labels)

    limit_ratio = minimum_num_labels / num_max_patches_per_image
    if limit_ratio > 1.0:
        target_num_positive_labels = int(round(minimum_num_labels / limit_ratio))
        target_num_negative_labels = int(round(minimum_num_labels / limit_ratio))
    else:
        target_num_positive_labels = minimum_num_labels
        target_num_negative_labels = minimum_num_labels

    # Shuffle lists together in order to select patches from different locations
    patches_labels = list(zip(patches, labels))
    random.shuffle(patches_labels)
    patches, labels = zip(*patches_labels)
    selected_patches_pos = []
    selected_labels_pos = []
    selected_patches_neg = []
    selected_labels_neg = []
    for patch, label in zip(patches, labels):
        if label == 1 and len(selected_labels_pos) < target_num_positive_labels:
            selected_patches_pos.append(patch)
            selected_labels_pos.append(label)
        elif label == 0 and len(selected_labels_neg) < target_num_negative_labels:
            selected_patches_neg.append(patch)
            selected_labels_neg.append(label)
        elif len(selected_labels_neg) >= target_num_negative_labels and len(selected_labels_pos) >= target_num_positive_labels:
            break
    selected_patches = selected_patches_pos
    selected_patches.extend(selected_patches_neg)
    selected_labels = selected_labels_pos
    selected_labels.extend(selected_labels_neg)
    return selected_patches, selected_labels


def select_data(image_dir, label_dir, image_extension, label_extension, num_images, num_max_patches_per_image = 10, base_resolution=200, square_data_size=0, context_window_ratio=2, min_patch_size=0):
    image_files =  glob.glob(image_dir + '/*' + image_extension)
    random.shuffle(image_files)
    selected_patches = []
    selected_labels = []
    for i in range(0, num_images):
        print('Generating patches for image {}/{}'.format(i+1, num_images))
        image = io.imread(image_files[i])
        # Check if image is corrupt or grayscale
        if len(image.shape) != 3:
            print('No RGB image: {}. Continuing.'.format(image_files[i]))
            continue
        label_image = io.imread(os.path.join(label_dir, os.path.basename(image_files[i])[:-len(image_extension)] + label_extension), as_grey=True)
        image = resize_image(image, base_resolution, base_resolution)
        label_image = resize_image(label_image, base_resolution, base_resolution)
        segments = segment_image(image, n_segments = 200, compactness=10, sigma=1)
        patches, labels = generate_patches_with_labels(image, label_image, segments, context_window_ratio = context_window_ratio, square_data_size=square_data_size, min_patch_size=min_patch_size)
        selected_patches_for_image, selected_labels_for_image = select_patches_50_to_50(patches, labels, num_max_patches_per_image=num_max_patches_per_image)
        selected_patches.extend(selected_patches_for_image)
        selected_labels.extend(selected_labels_for_image)
    print('Selected total number of patches: {}'.format(len(selected_patches)))
    return selected_patches, selected_labels


if __name__ == '__main__':
    relative_image_dir = '../../../../data/processed/cupv2/train_images'
    image_dir = os.path.join(os.path.dirname(__file__), relative_image_dir)
    relative_label_dir = '../../../../data/processed/cupv2/train_labels'
    label_dir = os.path.join(os.path.dirname(__file__), relative_label_dir)
    patches, labels = select_data(image_dir, label_dir, '.jpg', '.png', 100, num_max_patches_per_image = 30, base_resolution = 200)

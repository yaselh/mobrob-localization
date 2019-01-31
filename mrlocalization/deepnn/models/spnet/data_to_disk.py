import os
from data_selector import select_data
import numpy as np
import skimage.io as io

# Select training data
print('Generating training data')
relative_train_image_dir = '../../../../data/processed/thurv1/train_images'
train_image_dir = os.path.join(os.path.dirname(__file__), relative_train_image_dir)
relative_train_label_dir = '../../../../data/processed/thurv1/train_labels'
train_label_dir = os.path.join(os.path.dirname(__file__), relative_train_label_dir)
train_patches, train_labels = select_data(train_image_dir, train_label_dir, '.jpg', '.png', 1000, num_max_patches_per_image = 15, base_resolution = 200, square_data_size=150, context_window_ratio=2, min_patch_size=50)
train_patches_np = np.asarray(train_patches)
train_labels_np = np.asarray(train_labels)
train_samples_count = len(train_labels)

relative_train_pos_dir = '../../../../data/processed/thur_patches_v1/train/pos'
train_pos_dir = os.path.join(os.path.dirname(__file__), relative_train_pos_dir)
relative_train_neg_dir = '../../../../data/processed/thur_patches_v1/train/neg'
train_neg_dir = os.path.join(os.path.dirname(__file__), relative_train_neg_dir)

print('Saving {} positive samples'.format(train_labels.count(1)))

for i in range(0, train_samples_count):
    #print(train_labels[i])
    if train_labels_np[i] == 1:
        io.imsave('{}/{}.png'.format(train_pos_dir, i), train_patches_np[i,:,:,:])
    else:
        io.imsave('{}/{}.png'.format(train_neg_dir, i), train_patches_np[i,:,:,:])

# Select validation data
print('Generating validation data')
relative_val_image_dir = '../../../../data/processed/thurv1/val_images'
val_image_dir = os.path.join(os.path.dirname(__file__), relative_val_image_dir)
relative_val_label_dir = '../../../../data/processed/thurv1/val_labels'
val_label_dir = os.path.join(os.path.dirname(__file__), relative_val_label_dir)
val_patches, val_labels = select_data(val_image_dir, val_label_dir, '.jpg', '.png', 200, num_max_patches_per_image = 15, base_resolution = 200, square_data_size=150, context_window_ratio=2, min_patch_size=50)
val_patches_np = np.asarray(val_patches)
val_labels_np = np.asarray(val_labels)
val_samples_count = len(val_labels)

relative_val_pos_dir = '../../../../data/processed/thur_patches_v1/val/pos'
val_pos_dir = os.path.join(os.path.dirname(__file__), relative_val_pos_dir)
relative_val_neg_dir = '../../../../data/processed/thur_patches_v1/val/neg'
val_neg_dir = os.path.join(os.path.dirname(__file__), relative_val_neg_dir)

print('Saving {} positive samples'.format(val_labels.count(1)))

for i in range(0, val_samples_count):
    if val_labels_np[i] == 1:
        io.imsave('{}/{}.png'.format(val_pos_dir, i), val_patches_np[i,:,:,:])
    else:
        io.imsave('{}/{}.png'.format(val_neg_dir, i), val_patches_np[i,:,:,:])

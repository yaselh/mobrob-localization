import glob
import os
from shutil import copyfile

thur15k_dir = 'C:/Users/Daniel/Code/datasets/thur15k/CoffeeMug/Src'
relative_image_output_dir = '../../../data/processed/patchesv3/images'
image_output_dir = os.path.join(os.path.dirname(__file__), relative_image_output_dir)
relative_label_output_dir = '../../../data/processed/patchesv3/labels'
label_output_dir = os.path.join(os.path.dirname(__file__), relative_label_output_dir)

image_files = glob.glob(thur15k_dir + '/*.jpg')

for image_file in image_files:
    print(image_file[:-3] + 'png')
    if os.path.isfile(image_file[:-3] + 'png'):
        copyfile(image_file, image_output_dir + '/' + os.path.basename(image_file))
        copyfile(image_file[:-3] + 'png', label_output_dir + '/' + os.path.basename(image_file)[:-3] + 'png')
    else:
        print('No label available...')

import glob
import os

clear_dir='D:/mobrob_datasets/combinedv2/train'

mask_files =  glob.glob(clear_dir + '/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9].png')

for file in mask_files:
    os.remove(file)

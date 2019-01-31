# Source: https://github.com/theduynguyen/Keras-FCN
import numpy as np
import random
import os
import argparse

import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam, SGD

import resnet50_model
from data_generator import seg_data_generator
from loss_func import fcn_xent, fcn_xent_nobg, pixel_acc, mean_acc
import utils

import sys
sys.path.append('../utils')
from model_loader import load_model

def parse_args():
	parser = argparse.ArgumentParser()

	# Set number of training images here
	parser.add_argument('-t', '--n_train_img', help='Number of train images',
						type=int,default=9189)

	# Set number of validation images here
	parser.add_argument('-v', '--n_val_img', help='Number of validation images',
						type=int,default=390)

	parser.add_argument('-e', '--epochs', help='Number of epochs',
						type=int, default=40)

	parser.add_argument('-g', '--gpu', help='Use GPU ID',
						type=int, default=0)

	parser.add_argument('-o', '--opt', help='Optimizer',
						default='SGD')

	parser.add_argument('-d', '--img_dir', help='Directory containing the images',
						default='../../../../data/processed/cupv2/')

	parser.add_argument('-lr', '--learning_rate', help='Initial learning rate',
						default=0.01)

	parser.add_argument('-mi', '--model_input', help='Init with model',
					default='')

	relative_model_output_path = '../../../../models/model'
	model_output_path = os.path.join(os.path.dirname(__file__), relative_model_output_path)
	parser.add_argument('-mo', '--model_output', help='Where to save the trained model?',
					default=model_output_path)

	parser.add_argument('-id', '--exp_id', help='Experiment id',
					default='000')

	return parser.parse_args()


######################################################################################

args = parse_args()

utils.config_tf()

# create experimental directory
model_output_dir = args.model_output + args.exp_id
if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)

# set vars
N_train_img = args.n_train_img
N_val_img = args.n_val_img
N_epochs = args.epochs
n_classes = 2
model_input = args.model_input

# create model
gpu = '/gpu:' + str(args.gpu)
with tf.device(gpu):
	if model_input:
		model = load_model(model_input, custom_objects={'Softmax4D': resnet50_model.Softmax4D,
		'fcn_xent_nobg': fcn_xent_nobg,
		'mean_acc': mean_acc})
		stride= resnet50_model.get_stride()
	else:
		model = resnet50_model.build_model(n_classes)
		stride = resnet50_model.get_stride()

# create data generators
train_img_dir = args.img_dir + 'train_images/'
train_label_dir = args.img_dir + 'train_labels/'
val_img_dir = args.img_dir + 'val_images/'
val_label_dir = args.img_dir + 'val_labels/'

img_list_train = os.listdir(train_img_dir)
img_list_train = img_list_train[:N_train_img]
random.shuffle(img_list_train)
img_list_val = os.listdir(val_img_dir)
img_list_val = img_list_val[:N_val_img]

train_gen = seg_data_generator(stride,n_classes,train_img_dir,train_label_dir,img_list_train)
val_gen = seg_data_generator(stride,n_classes,val_img_dir,val_label_dir,img_list_val)

# callbacks
filepath = model_output_dir + '/best_model.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
							save_best_only=True, mode='min')


plateau = ReduceLROnPlateau(patience=5)

callbacks_list = [checkpoint, plateau]


learning_rate = float(args.learning_rate)
if args.opt == 'Adam':
	opt = Adam(lr=learning_rate)
elif args.opt == 'SGD':
	opt = SGD(lr=learning_rate, momentum=0.9)
elif args.opt == 'SGD_Aggr':
	opt = SGD(lr=learning_rate, momentum=0.99)

model.compile(optimizer = opt,loss = fcn_xent_nobg, metrics=[mean_acc])

print(model.summary())

# Transform parameters to keras 2
batch_size = 1
samples_per_epoch = N_train_img
steps_per_epoch = int(samples_per_epoch/batch_size)
model.fit_generator(train_gen,
					steps_per_epoch=steps_per_epoch, epochs=N_epochs,
					validation_data = val_gen,validation_steps = N_val_img,
					callbacks=callbacks_list,verbose=1)

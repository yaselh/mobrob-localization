# Source: https://github.com/theduynguyen/Keras-FCN
import os

import keras.backend as K
from keras.engine import Layer

from keras.layers import Input, Dropout, merge
from keras.layers.convolutional import Convolution2D, UpSampling2D, ZeroPadding2D, Cropping2D, Deconvolution2D
from keras.layers.core import Activation

from keras.applications.resnet50 import ResNet50
from keras.models import Model

import numpy as np

from loss_func import fcn_xent_nobg, mean_acc

file_dir = os.path.dirname(__file__)
relative_loader_dir = '../utils'
loader_dir = os.path.join(file_dir, relative_loader_dir)

import sys
sys.path.append(loader_dir)
from model_loader import load_model

class Softmax4D(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape

    def compute_output_shape(self, input_shape):
        return input_shape

def bilinear_interpolation(w):
	frac = w[0].shape[0]
	n_classes = w[0].shape[-1]
	w_bilinear = np.zeros(w[0].shape)

	for i in range(n_classes):
		w_bilinear[:,:,i,i] = 1.0/(frac*frac) * np.ones((frac,frac))

	return w_bilinear

def build_model(n_classes):
	# load ResNet
	input_tensor = Input(shape=(None, None, 3))
	base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

	# add classifier
	x = base_model.get_layer('activation_49').output
	x = Dropout(0.5)(x)
	x = Convolution2D(n_classes, (1, 1), name = 'pred_32',kernel_initializer='zero', padding='valid')(x)

	# add upsampler
	stride = 32
	x = UpSampling2D(size=(stride,stride))(x)
	x = Convolution2D(n_classes, (5, 5),name = 'pred_32s', kernel_initializer = 'zero', padding = 'same')(x)
	x = Softmax4D(axis=-1)(x)

	model = Model(inputs=base_model.input,outputs=x)

	# create bilinear interpolation
	w = model.get_layer('pred_32s').get_weights()
	model.get_layer('pred_32s').set_weights([bilinear_interpolation(w), w[1]])

	# fine-tune
	train_layers = ['pred_32',
					'pred_32s'

					'bn5c_branch2c',
					'res5c_branch2c',
					'bn5c_branch2b',
					'res5c_branch2b',
					'bn5c_branch2a',
					'res5c_branch2a',

					'bn5b_branch2c',
					'res5b_branch2c',
					'bn5b_branch2b',
					'res5b_branch2b',
					'bn5b_branch2a',
					'res5b_branch2a',

					'bn5a_branch2c',
					'res5a_branch2c',
					'bn5a_branch2b',
					'res5a_branch2b',
					'bn5a_branch2a',
					'res5a_branch2a']

	for l in model.layers:
		if l.name in train_layers:
			l.trainable = True
		else :
			l.trainable = False

	return model

def get_stride():
    stride = 32
    return stride

def save_architecture(n_classes, save_path):
    model = build_model(n_classes)
    json_string = model.to_json()
    with open(save_path, 'w') as text_file:
        text_file.write(json_string)

def load_model_from_file_or_url(path):
    model = load_model(path, custom_objects={'Softmax4D': Softmax4D,
    'fcn_xent_nobg': fcn_xent_nobg,
    'mean_acc': mean_acc})
    return model


# If executed as main the script saves its architecture as json
if __name__ == '__main__':
    save_architecture(n_classes = 2, save_path ='architecture_2_classes.json')

# Source: https://github.com/theduynguyen/Keras-FCN
import numpy as np
import os

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

import skimage.transform
import skimage.color
import skimage.io
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw

def pad_image(img,img_size) :
	max_dim = np.argmax(img.shape)
	min_dim = 1 - max_dim

	#resize the largest dim to img_size
	#if img.shape[max_dim] >= img_size:
	resize_factor = np.float(img_size) / np.float(img.shape[max_dim])
	new_min_dim_size = np.round( resize_factor * np.float(img.shape[min_dim]) )
	if len(img.shape) == 3:
		new_size = [img_size,img_size,3]
	else:
		new_size = [img_size, img_size, 1]
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
	if img.shape[2] == 1:
		img = img[:,:,0]
	return img

def seg_data_generator(stride,n_classes,img_dir,label_dir,img_list,preprocess = True):
	while 1:
		LUT = np.eye(n_classes)

		for img_id in img_list:

			# load image
			img_path = img_dir + img_id
			x = skimage.io.imread(img_path)

			# load label
			label_path = label_dir + img_id[:-3] + 'png'
			y = skimage.io.imread(label_path) # interprets the image as a colour image

			#only yield is the images exist
			is_img = type(x) is np.ndarray and type(y) is np.ndarray
			not_empty = len(x.shape) > 0 and len(y.shape) > 0

			if is_img and not_empty:
				#deal with gray value images
				if len(x.shape) == 2:
					x = skimage.color.gray2rgb(x)

				# only take one channel
				if len(y.shape) > 2:
					y = y[...,0]

				# treat binary images
				if np.max(y) == 255:
					y = np.clip(y,0,1)

				# COnvert to common input format
				x = pad_image(x, 224)
				y = pad_image(y, 224)

				# crop if image dims do not match stride
				w_rest = x.shape[0] % stride
				h_rest = x.shape[1] % stride

				if w_rest > 0:
					w_crop_1 = int(np.round(w_rest / 2))
					w_crop_2 = int(w_rest - w_crop_1)

					x = x[w_crop_1:-w_crop_2,:,:]
					y = y[w_crop_1:-w_crop_2,:]
				if h_rest > 0:
					h_crop_1 = int(np.round(h_rest / 2))
					h_crop_2 = int(h_rest - h_crop_1)


					x = x[:,h_crop_1:-h_crop_2,:]
					y = y[:,h_crop_1:-h_crop_2]


				#fig=plt.figure(figsize=(8, 8))
				#img = np.random.randint(10, size=(224,224))
				#columns = 2
				#rows = 1
				#fig.add_subplot(121)
				#show1 = plt.imshow(x)
				#fig.add_subplot(122)
				#show2 = plt.imshow(y)
				#plt.show()

				# prepare for NN
				x = np.array(x,dtype='float')
				x = x[np.newaxis,...]

				if preprocess == True:
					x = preprocess_input(x)
				y = LUT[y]
				y = y[np.newaxis,...] # make it a 4D tensor

				yield x, y

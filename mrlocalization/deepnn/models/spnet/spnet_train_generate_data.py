import os
from spnet_model_simple import build_model
from data_selector import select_data
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


# Select training data
print('Generating training data')
relative_train_image_dir = '../../../../data/processed/cupv2/train_images'
train_image_dir = os.path.join(os.path.dirname(__file__), relative_train_image_dir)
relative_train_label_dir = '../../../../data/processed/cupv2/train_labels'
train_label_dir = os.path.join(os.path.dirname(__file__), relative_train_label_dir)
train_patches, train_labels = select_data(train_image_dir, train_label_dir, '.jpg', '.png', 2000, num_max_patches_per_image = 30, base_resolution = 200, square_data_size=150)
train_patches_np = np.asarray(train_patches)
train_labels_np = np.asarray(train_labels)
train_samples_count = len(train_labels)


# Select validation data
print('Generating validation data')
relative_val_image_dir = '../../../../data/processed/cupv2/val_images'
val_image_dir = os.path.join(os.path.dirname(__file__), relative_val_image_dir)
relative_val_label_dir = '../../../../data/processed/cupv2/val_labels'
val_label_dir = os.path.join(os.path.dirname(__file__), relative_val_label_dir)
val_patches, val_labels = select_data(val_image_dir, val_label_dir, '.jpg', '.png', 390, num_max_patches_per_image = 30, base_resolution = 200, square_data_size=150)
val_patches_np = np.asarray(train_patches)
val_labels_np = np.asarray(train_labels)
val_samples_count = len(val_labels)

# Setup generators
batch_size = 16

train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow(train_patches_np, train_labels_np,
        batch_size=batch_size)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow(val_patches_np, val_labels_np, batch_size=batch_size)

model = build_model()
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['binary_accuracy'])

model.summary()

checkpointer = ModelCheckpoint(filepath='best_model.hdf5', verbose=1, save_best_only=True)

model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples_count // batch_size,
        epochs=50,
        validation_data=val_generator,
        validation_steps=val_samples_count // batch_size,
        callbacks=[checkpointer])

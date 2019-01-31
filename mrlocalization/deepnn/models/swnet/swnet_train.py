import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import glob

from swnet_model_simple5 import build_model

relative_train_dir = '../../../../data/processed/cup_bboxv3/train'
train_dir = os.path.join(os.path.dirname(__file__), relative_train_dir)
train_samples_count = len(glob.glob(train_dir + '/pos/*.png')) + len(glob.glob(train_dir + '/neg/*.png'))

relative_val_dir = '../../../../data/processed/cup_bboxv3/val'
val_dir = os.path.join(os.path.dirname(__file__), relative_val_dir)
val_samples_count = len(glob.glob(val_dir + '/pos/*.png')) + len(glob.glob(val_dir + '/neg/*.png'))



# Setup generators
batch_size = 16
train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=45.0,
        zoom_range=(1.0, 1.4),
        channel_shift_range = 0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_dir,  # this is the target directory
        batch_size=batch_size,
        class_mode='binary',
        target_size=(75,75))

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        val_dir,
        batch_size=batch_size,
        class_mode='binary',
        target_size=(75,75))

model = build_model()
optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['binary_accuracy'])

checkpointer = ModelCheckpoint(filepath='best_model.hdf5', verbose=1, save_best_only=True)

model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples_count // batch_size,
        epochs=100,
        validation_data=val_generator,
        validation_steps=val_samples_count // batch_size,
        callbacks=[checkpointer])

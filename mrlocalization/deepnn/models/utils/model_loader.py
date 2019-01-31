import os
from os.path import expanduser
import keras

def load_model(model_path, custom_objects = None):
    model = None
    if model_path.startswith('http'):
        home = expanduser("~")
        temp_dir_path = os.path.join(home, 'keras_model_temp')
        temp_file_path = os.path.join(temp_dir_path, 'temp.hdf5')
        if not os.path.exists(temp_dir_path):
            os.makedirs(temp_dir_path)
        keras.utils.get_file(fname=temp_file_path, origin=model_path)
        model = keras.models.load_model(temp_file_path, custom_objects)
        os.remove(temp_file_path)
        os.rmdir(temp_dir_path)
    else:
        model = keras.models.load_model(model_path, custom_objects)
    return model

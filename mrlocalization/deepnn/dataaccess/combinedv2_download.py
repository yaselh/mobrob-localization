import os
import keras
import zipfile

relative_dataset_dir = '../../../data/processed/combinedv2'
dataset_dir = os.path.join(os.path.dirname(__file__), relative_dataset_dir)

def download_to_folder(target_name, download_url):
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    target_path = os.path.abspath(dataset_dir + '/' + target_name)
    if not os.path.isfile(target_path):
        keras.utils.get_file(fname=target_path, origin=download_url)
        print('Unzipping files...')
        zip_ref = zipfile.ZipFile(target_path, 'r')
        zip_ref.extractall(os.path.dirname(target_path))
        zip_ref.close()


def download_training_data():
    download_to_folder('train_images.zip', 'http://cloud.dk-s.de/datasets/combinedv2/train_images.zip')

def download_training_labels():
    download_to_folder('train_labels.zip', 'http://cloud.dk-s.de/datasets/combinedv2/train_labels.zip')

def download_validation_data():
    download_to_folder('val_images.zip', 'http://cloud.dk-s.de/datasets/combinedv2/val_images.zip')

def download_validation_labels():
    download_to_folder('val_labels.zip', 'http://cloud.dk-s.de/datasets/combinedv2/val_labels.zip')

def download_labels_txt():
    target_path = os.path.abspath(dataset_dir + '/labels.txt')
    download_url= 'http://cloud.dk-s.de/datasets/combinedv2/labels.txt'
    keras.utils.get_file(fname=target_path, origin=download_url)

if __name__=='__main__':
    download_validation_data()
    download_validation_labels()
    download_training_data()
    download_training_labels()
    download_labels_txt()

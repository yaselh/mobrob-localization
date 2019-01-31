# This script prepares an ec2 instance for neural network training
from subprocess import call
import os


# Install dependencies
print('Installing dependencies...')
relative_requirements_path = '../../../requirements_no_version.txt'
requirements_path = os.path.join(os.path.dirname(__file__), relative_requirements_path)
dependeny_install_call = 'pip3 install -r ' + relative_requirements_path + ' --user'
call_array = dependeny_install_call.split(' ')
call(call_array)

# Download datasets
print('Downloading datasets...')
relative_script_path = '../dataaccess/combinedv8_download.py'
script_path = os.path.join(os.path.dirname(__file__), relative_script_path)
dataset_install_call = 'python3 ' + script_path
call_array = dataset_install_call.split(' ')
call(call_array)

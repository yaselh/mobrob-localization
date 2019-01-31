from subprocess import call
import os

#Note: In order for this to work the putty directory that includes pscp has to be in path
host='ec2-54-171-251-187.eu-west-1.compute.amazonaws.com'
user = 'ec2-user'
key_file = 'C:/Users/Daniel/aws_keys/mobrob-2.ppk'
remote_path = '/home/ec2-user/mobrob-localization/models/model000/best_model.hdf5'
relative_local_path = '../../../models/resnet50/combinedv8/model_epoch_13.hdf5'
local_path = os.path.join(os.path.dirname(__file__), relative_local_path)
call_string = 'pscp -i ' + key_file + ' ' + user + '@' + host + ':' + remote_path + ' ' + local_path
call_array = call_string.split(' ')
call(call_array)

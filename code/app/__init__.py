import os

from flask import Flask

application = Flask(__name__)
application.secret_key = os.urandom(24)

from app.download_weights import *

weight_path = 'app/models/frcnn_detector/weights'
weight_name = 'model_frcnn_vgg2.hdf5'

# This is for testing, comment out on actual deploy to always ensure download
if os.path.isfile(os.path.join(weight_path, weight_name)):
    print('Weight is already downloaded.')
else:
    print('Downloading weights...')
    assert os.path.isdir(
        weight_path), f'{weight_path} does not exist, please create'
    download_frcnn_weights(os.path.join(weight_path, weight_name))
    print('Done!')

from app import routes

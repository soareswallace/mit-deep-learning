import tensorflow as tf
from deepLab import DeepLabModel
import os
from visualization_model import vis_segmentation, run_visualization_on_static_image

from io import BytesIO
import tempfile
from six.moves import urllib

import matplotlib
from tqdm import tqdm
import IPython
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

import warnings

SAMPLE_IMAGE = '../tutorial_driving_scene_segmentation/mit_driveseg_sample.png'
MODEL_NAME = 'mobilenetv2_coco_cityscapes_trainfine'
#MODEL_NAME = 'xception65_cityscapes_trainfine'
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_cityscapes_trainfine':
        'deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz',
    'xception65_cityscapes_trainfine':
        'deeplabv3_cityscapes_train_2018_02_06.tar.gz',
}
_TARBALL_NAME = 'deeplab_model.tar.gz'


#Model Configuration
model_dir = tempfile.mkdtemp()
tf.gfile.MakeDirs(model_dir)
download_path = os.path.join(model_dir, _TARBALL_NAME)
print('downloading model, this might take a while...')
urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)
print('download completed! loading DeepLab model...')
MODEL = DeepLabModel(download_path)
print('model loaded successfully!')
print('running deeplab on the sample image...')

run_visualization_on_static_image(SAMPLE_IMAGE, MODEL)

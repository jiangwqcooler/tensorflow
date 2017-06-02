# _*_ coding:utf-8 _*_

"""
@author jiangwenqiang
@date 2017/05/22
"""

"""
vgg net parameters config
"""

# training times
TRAIN_STEPS = 20000

# input image size
IMAGE_SIZE = 224

IMAGE_CHANNELS = 3

# snapshoot step
STEP_MODEL_SAVE = 10000

# save image cache dir
IMAGE_CACHE_DIR = '../cache/'

TFRECORD_DIR = '../record/'

# save model dir
MODEL_DIR = '../models/'

DATA_DIR = '../data/'

LOG_DIR = '../logs/'

# percentage of dataset
VALIDATION_PERCENTAGE = 0.1

# percentage of dataset
TEST_PERCENTAGE = 0.1

LEARNING_RATE_BASE = 0.00001

BATCH_SIZE = 64

LEARNING_RATE_DECAY = 0.99

REGULARAZTION_RATE = 0.0001

MOVING_AVERAGE_DECAY = 0.99

USE_MODEL = False

WEIGHT_REGULARIZER = True

NUM_LABELS = 5

COMPRESSION_METHOD = 0

SHUFFLE_CAPACITY = 10000

MODEL_NAME = 'vgg16-model.ckpt'

FINETUNE = False

IMAGE_TRANSPOSE = True

IMAGE_BRIGHTNESS = True

IMAGE_HUE = True

IMAGE_SATURATION = True

IMAGE_RANDOM_CROP = True

DATASET_NAME = '4d_dataset'





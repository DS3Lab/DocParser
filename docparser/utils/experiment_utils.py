import argparse
import time

import git
import keras
import numpy as np
from mrcnn.config import Config as mrcnn_config


# https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit/43186440
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class DocparserDefaultConfig(mrcnn_config):
    """Configuration for training on the docs dataset.
    Derives from the base Config class and overrides values specific
    to the docs dataset.
    """
    # Give the configuration a recognizable name

    # GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 28

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = None
    IMAGE_MAX_DIM = 1024  # pad to max dim (or shrink to fit it)
    IMAGE_RESIZE_MODE = "square"

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    MAX_GT_INSTANCES = 100
    DETECTION_MAX_INSTANCES = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 2000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 200

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 0.5
    }


def create_config_log_dict(config, dataset_train, dataset_val, args):
    mrcnn_config = dict()
    for a in dir(config):
        if not a.startswith("__") and not callable(getattr(config, a)):
            config_obj = getattr(config, a)
            if isinstance(config_obj, np.ndarray):
                mrcnn_config[a] = config_obj.tolist()
            else:
                mrcnn_config[a] = config_obj
    train_images = {str(img_id): dataset_train.image_info[img_id]['path'] for img_id in dataset_train.image_ids}
    train_images2 = []
    if dataset_val is not None:
        val_images = {str(img_id): dataset_val.image_info[img_id]['path'] for img_id in dataset_val.image_ids}
    else:
        val_images = None
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    config_log = {'mrcnn_config': mrcnn_config, 'args': vars(args), 'dataset_train': train_images,
                  'dataset_val': val_images, 'git': {'sha': sha}}
    return config_log


def get_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on a Document Annotation dataset.')
    parser.add_argument("command",
                        help="'train' or 'evaluate' on dataset")
    parser.add_argument('--train-dataset', required=True,
                        help='Directory of the Document Annotation dataset')
    parser.add_argument('--train-version', required=True,
                        help='Annotation file version')
    parser.add_argument('--train-version2',
                        help='Optional second train annotation file version')
    parser.add_argument('--val-dataset', required=True,
                        help='Directory of the Document Annotation dataset')
    parser.add_argument('--val-version', required=True,
                        help='Annotation file version (default=man)')
    parser.add_argument('--epochs-offset', type=int, default=0,
                        help="Number of total epochs")
    parser.add_argument('--epochs1', type=int, default=20,
                        help="Number of total epochs")
    parser.add_argument('--epochs2', type=int, default=60,
                        help="Number of total epochs in stage 2")
    parser.add_argument('--epochs3', type=int, default=80,
                        help="Number of total epochs in stage 3")
    parser.add_argument('--gpu-count', type=int, default=1,
                        help="Number gpus to use")
    parser.add_argument('--augmentation', action='store_true',
                        help="Image augmentation (right/left flip 50% of the time)")
    parser.add_argument('--chargrid', action='store_true',
                        help="use chargrid images")
    parser.add_argument('--only-multicells', action='store_true',
                        help="only use table cells that span more than one column or row")
    parser.add_argument('--only-labelcells', action='store_true',
                        help="only use table cells that span more than one column or row")

    parser.add_argument('--train-rois-per-image', type=int, default=200,
                        help="Number of ROIs per image to feed to classifier/mask heads")
    parser.add_argument('--detection-max-instances', type=int, default=100,
                        help="Max number of final detections")
    parser.add_argument('--max-gt-instances', type=int, default=100,
                        help="Maximum number of ground truth instances to use in one image")
    parser.add_argument('--steps-per-epoch', type=int, default=1000,
                        help="Number of training steps per epoch")
    parser.add_argument('--validation-steps', type=int, default=50,
                        help="Number of validation steps to run at the end of every training epoch")
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--resize-mode', type=str, default='square',
                        help="resizing method for images")

    parser.add_argument('--classes', type=str, nargs='+',
                        help="Specify the classes to train on", required=True)
    parser.add_argument('--model', required=True,
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=True,
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--name',
                        help='Name')
    parser.add_argument('--subset-sample', action='store_true',
                        help='Only use a smaller subset of the images in the dataset')
    parser.add_argument('--manualseed', type=int,
                        help='manual random seed for subset selection')
    parser.add_argument('--numimgs', type=int,
                        help='number of images to be sampled randomly with manual random seed')
    parser.add_argument('--nms-threshold', type=float,
                        help='iou threshold value for detection non-maximum supression')
    args = parser.parse_args()

    print("Args:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    return args

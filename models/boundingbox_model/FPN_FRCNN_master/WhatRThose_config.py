"""
Faster R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import cv2
import os
import sys
import json
import datetime
import matplotlib.pyplot as plt 
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Faster RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from frcnn.config import Config
from frcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class run1(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "test6"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + test2

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 400

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.5
    
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (64, 128, 256, 512, 1024)
    
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    LOSS_WEIGHTS = {
        "rpn_class_loss": 2.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
        }
    
    
############################################################
#  Dataset
############################################################

class Dataset(utils.Dataset):

    def load_data(self, dataset_dir, subset):
        """Load a subset of the test2 dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class(dataset_dir.split('/')[-1].split('.')[0], 1, "shoe")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
#         dataset_dir = os.path.join(dataset_dir, subset)

        found_bg = False
        all_imgs = {}

        classes_count = {}

        class_mapping = {}

        visualise = True

        with open(dataset_dir,'r') as f:

            print('Parsing annotation {} files'.format(subset))

            for line in f:
                line_split = line.strip().split(',')
                (filename,x1,y1,x2,y2,class_name) = line_split

                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if class_name not in class_mapping:
                    if class_name == 'bg' and found_bg == False:
                        print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                        found_bg = True
                    class_mapping[class_name] = len(class_mapping)

                if filename not in all_imgs:
                    all_imgs[filename] = {}
                    img = plt.imread(filename)
                    (rows,cols) = img.shape[:2]
                    all_imgs[filename]['id'] = filename.split('/')[-1]
                    all_imgs[filename]['path'] = filename                
                    all_imgs[filename]['source'] = dataset_dir.split('/')[-1].split('.')[0]
                    all_imgs[filename]['width'] = cols
                    all_imgs[filename]['height'] = rows
                    all_imgs[filename]['bboxes'] = []
                    all_imgs[filename]['imageset'] = subset

                all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)), 'y2': int(float(y2))})


            all_data = []
            for key in all_imgs:
                all_data.append(all_imgs[key])

            # make sure the bg class is last in the list
            if found_bg:
                if class_mapping['bg'] != len(class_mapping) - 1:
                    key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
                    val_to_switch = class_mapping['bg']
                    class_mapping['bg'] = len(class_mapping) - 1
                    class_mapping[key_to_switch] = val_to_switch

#             return all_data, classes_count, class_mapping  
        print('{} {} files'.format(len(all_data),subset))
        self.image_info = all_data
        
        
#     def load_mask(self, image_id):
    def load_class(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # one class ID only, we return an array of 1s
        image_info = self.image_info[image_id]['bboxes']
        return np.ones([len(image_info)],dtype=np.int32)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
#         return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        
        
        
    

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "test2":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = test2Dataset()
    dataset_train.load_test2(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = test2Dataset()
    dataset_val.load_test2(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')
    
    
# ############################################################
# #  Training
# ############################################################

# if __name__ == '__main__':
#     import argparse

#     # Parse command line arguments
#     parser = argparse.ArgumentParser(
#         description='Train Mask R-CNN to detect balloons.')
#     parser.add_argument("command",
#                         metavar="<command>",
#                         help="'train' or 'splash'")
#     parser.add_argument('--dataset', required=False,
#                         metavar="/path/to/balloon/dataset/",
#                         help='Directory of the Balloon dataset')
#     parser.add_argument('--weights', required=True,
#                         metavar="/path/to/weights.h5",
#                         help="Path to weights .h5 file or 'coco'")
#     parser.add_argument('--logs', required=False,
#                         default=DEFAULT_LOGS_DIR,
#                         metavar="/path/to/logs/",
#                         help='Logs and checkpoints directory (default=logs/)')
#     parser.add_argument('--image', required=False,
#                         metavar="path or URL to image",
#                         help='Image to apply the color splash effect on')
#     parser.add_argument('--video', required=False,
#                         metavar="path or URL to video",
#                         help='Video to apply the color splash effect on')
#     args = parser.parse_args()

#     # Validate arguments
#     if args.command == "train":
#         assert args.dataset, "Argument --dataset is required for training"
#     elif args.command == "splash":
#         assert args.image or args.video,\
#                "Provide --image or --video to apply color splash"

#     print("Weights: ", args.weights)
#     print("Dataset: ", args.dataset)
#     print("Logs: ", args.logs)

#     # Configurations
#     if args.command == "train":
#         config = BalloonConfig()
#     else:
#         class InferenceConfig(BalloonConfig):
#             # Set batch size to 1 since we'll be running inference on
#             # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#             GPU_COUNT = 1
#             IMAGES_PER_GPU = 1
#         config = InferenceConfig()
#     config.display()

#     # Create model
#     if args.command == "train":
#         model = modellib.MaskRCNN(mode="training", config=config,
#                                   model_dir=args.logs)
#     else:
#         model = modellib.MaskRCNN(mode="inference", config=config,
#                                   model_dir=args.logs)

#     # Select weights file to load
#     if args.weights.lower() == "coco":
#         weights_path = COCO_WEIGHTS_PATH
#         # Download weights file
#         if not os.path.exists(weights_path):
#             utils.download_trained_weights(weights_path)
#     elif args.weights.lower() == "last":
#         # Find last trained weights
#         weights_path = model.find_last()
#     elif args.weights.lower() == "imagenet":
#         # Start from ImageNet trained weights
#         weights_path = model.get_imagenet_weights()
#     else:
#         weights_path = args.weights

#     # Load weights
#     print("Loading weights ", weights_path)
#     if args.weights.lower() == "coco":
#         # Exclude the last layers because they require a matching
#         # number of classes
#         model.load_weights(weights_path, by_name=True, exclude=[
#             "mrcnn_class_logits", "mrcnn_bbox_fc",
#             "mrcnn_bbox", "mrcnn_mask"])
#     else:
#         model.load_weights(weights_path, by_name=True)

#     # Train or evaluate
#     if args.command == "train":
#         train(model)
#     elif args.command == "splash":
#         detect_and_color_splash(model, image_path=args.image,
#                                 video_path=args.video)
#     else:
#         print("'{}' is not recognized. "
#               "Use 'train' or 'splash'".format(args.command))
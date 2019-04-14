from config import Config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from model import *
from utils import * 
import time     
import os
import pickle


# Hardcoded Paths, These can be changed to be overidden from command line in the future
input_path = 'input_images'
weight_path = 'weights/model_frcnn_vgg.hdf5'
config_path = 'weights/model_vgg_config.pickle'
save_imgs_path = 'output_images'

# Read in Config for Testing 
with open(config_path, 'rb') as f_in:
    C = pickle.load(f_in)

# Turn off any data augmentation at test time
C.model_path = weight_path
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False
class_mapping = C.class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}

# Gather Model Inputs 
num_features = 512
input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# Define Model Layers 
shared_layers = nn_base(img_input, trainable=True)
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = rpn_layer(shared_layers, num_anchors)
classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

# Build Models 
model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)
model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

# Grab Image from path
test_imgs = os.listdir(input_path)

# Predict
bbox_threshold = 0.95

final_predictions = []

T = {}
P = {}
mAPs = []
for test_img in test_imgs:
    if test_img[0] == '.':
        continue
    filepath = os.path.join(input_path, test_img)
    img_o = cv2.imread(filepath)
    img = img_o.copy()
    X, ratio = format_img(img, C)

    _, fx, fy = format_img_map(img, C)

    # Change X (img) shape from (1, channel, height, width) to (1, height, width, channel)
    X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)

    R = rpn_to_roi(Y1, Y2, C, overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(
            R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        # Calculate all classes' bboxes coordinates on resized image (300, 400)
        # Drop 'bg' classes bboxes
        for ii in range(P_cls.shape[1]):

            # If class name is 'bg', continue
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            # Get class name
            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append(
                [16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []
    real_dets = []

    # if no predictions
    if bboxes == {}:
        real_dets.append(
            {'x1': 0, 'y1': 0, 'x2': img.shape[0], 'y2': img.shape[1], 'path': test_img})

    for key in bboxes:
        bbox = np.array(bboxes[key])

        # Apply non-max-suppression on final bboxes to get the output bounding boxe
        new_boxes, new_probs = non_max_suppression_fast(
            bbox, np.array(probs[key]), overlap_thresh=0.2)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]
            det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,
                   'class': key, 'prob': new_probs[jk]}
            all_dets.append(det)

            # Calculate real coordinates on original image
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(
                ratio, x1, y1, x2, y2)
            real_det = {'x1': real_x1, 'x2': real_x2, 'y1': real_y1, 'y2': real_y2, 'prob': new_probs[jk],
                        'path': test_img.split('.')[0]+'('+str(jk)+')'+'.jpg'}
            real_dets.append(real_det)

    final_predictions.append([img_o, real_dets])


# Crop images and save
save_crop(final_predictions, save_imgs_path)
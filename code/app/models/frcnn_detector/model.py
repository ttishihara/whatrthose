import tensorflow as tf
from keras import backend as K
# from keras.layers import Layers.Flatten, Layers.Dense, Layers.Input, Layers.Conv2D, Layers.MaxPooling2D, Layers.Dropout, \
    # Layers.TimeDistributed
from keras.engine import Layer, Layers.InputSpec
import keras.layers as Layers
import numpy as np


def nn_base(Layers.Input_tensor=None, trainable=False):
    Layers.Input_shape = (None, None, 3)

    if Layers.Input_tensor is None:
        img_Layers.Input = Layers.Input(shape=Layers.Input_shape)
    else:
        if not K.is_keras_tensor(Layers.Input_tensor):
            img_Layers.Input = Layers.Input(tensor=Layers.Input_tensor, shape=Layers.Input_shape)
        else:
            img_Layers.Input = Layers.Input_tensor

    bn_axis = 3

    # Block 1
    x = Layers.Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1')(img_Layers.Input)
    x = Layers.Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv2')(x)
    x = Layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Layers.Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1')(x)
    x = Layers.Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv2')(x)
    x = Layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Layers.Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1')(x)
    x = Layers.Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv2')(x)
    x = Layers.Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv3')(x)
    x = Layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Layers.Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1')(x)
    x = Layers.Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv2')(x)
    x = Layers.Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv3')(x)
    x = Layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Layers.Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1')(x)
    x = Layers.Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv2')(x)
    x = Layers.Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv3')(x)
    # x = Layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    return x


def rpn_layer(base_layers, num_anchors):
    """Create a rpn layer
        Step1: Pass through the feature map from base layer to a 3x3 512
                channels convolutional layer. Keep the padding 'same' to
                preserve the feature map's size
        Step2: Pass the step1 to two (1,1) convolutional layer to replace the
                fully connected layer classification layer: num_anchors
                (9 in here) channels for 0, 1 sigmoid activation output
                regression layer: num_anchors*4 (36 in here) channels for
                computing the regression of bboxes with linear activation
    Args:
        base_layers: vgg in here
        num_anchors: 9 in here

    Returns:
        [x_class, x_regr, base_layers]
        x_class: classification for whether it's an object
        x_regr: bboxes regression
        base_layers: vgg in here
    """
    x = Layers.Conv2D(512, (3, 3), padding='same', activation='relu',
               kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Layers.Conv2D(num_anchors, (1, 1), activation='sigmoid',
                     kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Layers.Conv2D(num_anchors * 4, (1, 1), activation='linear',
                    kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D Layers.Inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual
    Recognition, K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7
            region.
        num_rois: number of regions of interest to be used
    # Layers.Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''

    def __init__(self, pool_size, num_rois, **kwargs):
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, Layers.Input_shape):
        self.nb_channels = Layers.Input_shape[0][3]

    def compute_output_shape(self, Layers.Input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size,\
               self.nb_channels

    def call(self, x, mask=None):
        assert (len(x) == 2)

        # x[0] is image with shape (rows, cols, channels)
        img = x[0]

        # x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        Layers.Input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # Resized roi of the image to pooling size (7x7)
            rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :],
                                        (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        # Might be (1, 4, 7, 7, 3)
        final_output = K.reshape(final_output, (
            1, self.num_rois, self.pool_size, self.pool_size,
            self.nb_channels))

        # permute_dimensions is similar to transpose
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def classifier_layer(base_layers, Layers.Input_rois, num_rois, nb_classes=4):
    """Create a classifier layer
    
    Args:
        base_layers: vgg
        Layers.Input_rois: `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
        num_rois: number of rois to be processed in one time (4 in here)

    Returns:
        list(out_class, out_regr)
        out_class: classifier layer output
        out_regr: regression layer output
    """

    Layers.Input_shape = (num_rois, 7, 7, 512)

    pooling_regions = 7

    # out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
    # num_rois (4) 7x7 roi pooling
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)(
        [base_layers, Layers.Input_rois])

    # Layers.MaxPooling2D the convlutional layer and connected to 2 FC and 2 Layers.Dropout
    out = Layers.TimeDistributed(Layers.Flatten(name='Layers.Flatten'))(out_roi_pool)
    out = Layers.TimeDistributed(Layers.Dense(4096, activation='relu', name='fc1'))(out)
    out = Layers.TimeDistributed(Layers.Dropout(0.5))(out)
    out = Layers.TimeDistributed(Layers.Dense(4096, activation='relu', name='fc2'))(out)
    out = Layers.TimeDistributed(Layers.Dropout(0.5))(out)

    # There are two output layer
    # out_class: softmax acivation function for classify the class name of the
    #           object
    # out_regr: linear activation function for bboxes coordinates regression
    out_class = Layers.TimeDistributed(
        Layers.Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
        name='Layers.Dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = Layers.TimeDistributed(Layers.Dense(4 * (nb_classes - 1), activation='linear',
                                     kernel_initializer='zero'),
                               name='Layers.Dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]


def rpn_to_roi(rpn_layer, regr_layer, C, use_regr=True, max_boxes=300,
               overlap_thresh=0.9):
    """Convert rpn layer to roi bboxes

    Args: (num_anchors = 9)
        rpn_layer: output layer for rpn classification 
            shape (1, feature_map.height, feature_map.width, num_anchors)
            Might be (1, 18, 25, 9) if resized image is 400 width and 300
        regr_layer: output layer for rpn regression
            shape (1, feature_map.height, feature_map.width, num_anchors)
            Might be (1, 18, 25, 36) if resized image is 400 width and 300
        C: config
        use_regr: Wether to use bboxes regression in rpn
        max_boxes: max bboxes number for non-max-suppression (NMS)
        overlap_thresh: If iou in NMS is larger than this threshold, drop the
            box

    Returns:
        result: boxes from non-max-suppression (shape=(300, 4))
            boxes: coordinates for bboxes (on the feature map)
    """
    regr_layer = regr_layer / C.std_scaling

    anchor_sizes = C.anchor_box_scales  # (3 in here)
    anchor_ratios = C.anchor_box_ratios  # (3 in here)

    assert rpn_layer.shape[0] == 1

    (rows, cols) = rpn_layer.shape[1:3]

    curr_layer = 0

    # A.shape = (4, feature_map.height, feature_map.width, num_anchors) 
    # Might be (4, 18, 25, 9) if resized image is 400 width and 300
    # A is the coordinates for 9 anchors for every point in the feature map 
    # => all 18x25x9=4050 anchors cooridnates
    A = np.zeros(
        (4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            # anchor_x = (128 * 1) / 16 = 8  => width of current anchor
            # anchor_y = (128 * 2) / 16 = 16 => height of current anchor
            anchor_x = (anchor_size * anchor_ratio[0]) / C.rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1]) / C.rpn_stride

            # curr_layer: 0~8 (9 anchors)
            # the Kth anchor of all position in the feature map (9th in total)
            regr = regr_layer[0, :, :,
                   4 * curr_layer:4 * curr_layer + 4]  # shape => (18, 25, 4)
            regr = np.transpose(regr, (2, 0, 1))  # shape => (4, 18, 25)

            # Create 18x25 mesh grid
            # For every point in x, there are all the y points and vice versa
            # X.shape = (18, 25)
            # Y.shape = (18, 25)
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

            # Calculate anchor position and size for each feature map point
            A[0, :, :, curr_layer] = X - anchor_x / 2  # Top left x coordinate
            A[1, :, :, curr_layer] = Y - anchor_y / 2  # Top left y coordinate
            A[2, :, :, curr_layer] = anchor_x  # width of current anchor
            A[3, :, :, curr_layer] = anchor_y  # height of current anchor

            # Apply regression to x, y, w and h if there is rpn regression
            # layer
            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer],
                                                       regr)

            # Avoid width and height exceeding 1
            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])

            # Convert (x, y , w, h) to (x1, y1, x2, y2)
            # x1, y1 is top left coordinate
            # x2, y2 is bottom right coordinate
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

            # Avoid bboxes drawn outside the feature map
            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols - 1,
                                                A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows - 1,
                                                A[3, :, :, curr_layer])

            curr_layer += 1

    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose(
        (1, 0))  # shape=(4050, 4)
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape(
        (-1))  # shape=(4050,)

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    # Find out the bboxes which is illegal and delete them from bboxes list
    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    # Apply non_max_suppression
    # Only extract the bboxes. Don't need rpn probs in the later process
    result = non_max_suppression_fast(all_boxes, all_probs,
                                      overlap_thresh=overlap_thresh,
                                      max_boxes=max_boxes)[0]

    return result


def apply_regr_np(X, T):
    """Apply regression layer to all anchors in one feature map

    Args:
        X: shape=(4, 18, 25) the current anchor type for all points in the
            feature map
        T: regression layer shape=(4, 18, 25)

    Returns:
        X: regressed position and size for current anchor
    """
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X


def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    # code used from here:
    # http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list

    # Process explanation:
    #   Step 1: Sort the probs list
    #   Step 2: Find the larget prob 'Last' in the list and save it to the pick
    #           list
    #   Step 3: Calculate the IoU with 'Last' box and other boxes in the list.
    #           If the IoU is larger than overlap_threshold, delete the box
    #           from list
    #   Step 4: Repeat step 2 and step 3 until there is no item in the probs
    #           list
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes    
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes 
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int / (area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(
                                                   overlap > overlap_thresh)[
                                                   0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data
    # type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs


def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)

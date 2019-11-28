# -*- coding: utf-8 -*-
'''
 Conversion functions for image viewer extension
'''

import cv2
import six
import numpy as np

import drawing

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def face_img_func(key, entry, viewer):
    # Image conversion
    img = entry['img'][0]   # Use only a first data in the batch
    assert(img.ndim == 3 and (img.shape[0] == 1 or img.shape[0] == 3))
    img = np.transpose(img, (1, 2, 0))
    img = img.copy()  # for safety
    img += 0.5  # [-0.5:0.5] -> [0:1]
    # Draw
    try:
        detection_raw = entry['detection'][0]
        detection = (detection_raw > 0.5)
        if 0.0 <= detection_raw <= 1.0:
            drawing.draw_detection(img, detection)

        landmark = entry['landmark'][0]
        visibility = entry['visibility'][0]
        landmark_color = (0, 1, 0) if detection == 1 else (0, 0, 1)
        drawing.draw_landmark(img, landmark, visibility, landmark_color, 0.5)

        pose = entry['pose'][0]
        drawing.draw_pose(img, pose)

        gender = entry['gender'][0]
        if 0.0 <= gender <= 1.0:
            gender = (gender > 0.5)
            drawing.draw_gender(img, gender)

    except KeyError:
        pass

    img = (img * 255).astype(np.uint8)
    caption = '{:02d}'.format(viewer.img_cnts[key])
    return {'img': img, 'cap': caption}


def weights_img_func(key, entry, viewer):
    data = entry['weights']
    assert(data.ndim == 4)
    img_cnt_max = viewer.img_cnt_max[key]

    res_data = list()

    # accumulate to 3 channels image
    for i in six.moves.range(min(data.shape[0], img_cnt_max)):
        img_shape = (3,) + data.shape[2:4]
        accum = np.zeros(img_shape, dtype=data.dtype)
        for ch in six.moves.range(data.shape[1]):
            accum[ch % 3] += data[i][ch]

        # normalize
        img = np.transpose(accum, (1, 2, 0))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        width = img.shape[0] * 15
        res_data.append({'img': img, 'width': width})

    return res_data
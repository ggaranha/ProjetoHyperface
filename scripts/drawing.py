# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math

import datasets

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())

# matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_agg as agg
except Exception as e:
    logger.error('Failed to import matplotlib')
    logger.error('[%s] %s', str(type(e)), str(e.args))
    exit()


def _draw_line(img, pt1, pt2, color, thickness=2):
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]), int(pt2[1]))
    cv2.line(img, pt1, pt2, color, int(thickness))


def _draw_circle(img, pt, color, radius=4, thickness=-1):
    pt = (int(pt[0]), int(pt[1]))
    cv2.circle(img, pt, radius, color, int(thickness))


def _draw_rect(img, rect, color, thickness=2):
    p1 = (int(rect[0]), int(rect[1]))
    p2 = (int(rect[0] + rect[2]), int(rect[1] + rect[3]))
    cv2.rectangle(img, p1, p2, color, thickness)


def _draw_cross(img, pt, color, size=4, thickness=2):
    p0 = (pt[0] - size, pt[1] - size)
    p1 = (pt[0] + size, pt[1] + size)
    p2 = (pt[0] + size, pt[1] - size)
    p3 = (pt[0] - size, pt[1] + size)
    _draw_line(img, p0, p1, color, thickness)
    _draw_line(img, p2, p3, color, thickness)





def draw_detection(img, detection, size=15):
    # Upper left
    pt = (size + 5, size + 5)
    if detection:
        _draw_circle(img, pt, (0, 0.7, 0), size, 5)
    else:
        _draw_cross(img, pt, (0, 0, 0.7), size, 5)


def draw_landmark(img, landmark, visibility, color, line_color_scale,
                  denormalize_scale=True):
    """  Draw AFLW 21 points landmark
        0|LeftBrowLeftCorner
        1|LeftBrowCenter
        2|LeftBrowRightCorner
        3|RightBrowLeftCorner
        4|RightBrowCenter
        5|RightBrowRightCorner
        6|LeftEyeLeftCorner
        7|LeftEyeCenter
        8|LeftEyeRightCorner
        9|RightEyeLeftCorner
        10|RightEyeCenter
        11|RightEyeRightCorner
        12|LeftEar
        13|NoseLeft
        14|NoseCenter
        15|NoseRight
        16|RightEar
        17|MouthLeftCorner
        18|MouthCenter
        19|MouthRightCorner
        20|ChinCenter
    """
    conn_list = [[0, 1], [1, 2], [3, 4], [4, 5],  # brow
                 [6, 7], [7, 8], [9, 10], [10, 11],  # eye
                 [13, 14], [14, 15], [13, 15],  # nose
                 [17, 18], [18, 19],  # mouse
                 [12, 20], [16, 20]]  # face contour

    if landmark.ndim == 1:
        landmark = landmark.reshape(int(landmark.shape[-1] / 2), 2)
    assert(landmark.shape[0] == 21 and visibility.shape[0] == 21)

    if denormalize_scale:
        h, w = img.shape[0:2]
        size = np.array([[w, h]], dtype=np.float32)
        landmark = landmark * size + size / 2

    # Line
    line_color = tuple(v * line_color_scale for v in color)
    for i0, i1 in conn_list:
        if visibility[i0] > 0.5 and visibility[i1] > 0.5:
            _draw_line(img, landmark[i0], landmark[i1], line_color, 2)

    # Point
    for pt, visib in zip(landmark, visibility):
        if visib > 0.5:
            _draw_circle(img, pt, color, 4, -1)
        else:
            _draw_circle(img, pt, color, 4, 1)


def _rotation_matrix(rad_x, rad_y, rad_z):
    # Generate the rotation matrix
    cosx, cosy, cosz = math.cos(rad_x), math.cos(rad_y), math.cos(rad_z)
    sinx, siny, sinz = math.sin(rad_x), math.sin(rad_y), math.sin(rad_z)
    rotz = np.array([[cosz, -sinz, 0],
                     [sinz, cosz, 0],
                     [0, 0, 1]], dtype=np.float32)
    roty = np.array([[cosy, 0, siny],
                     [0, 1, 0],
                     [-siny, 0, cosy]], dtype=np.float32)
    rotx = np.array([[1, 0, 0],
                     [0, cosx, -sinx],
                     [0, sinx, cosx]], dtype=np.float32)
    return rotx.dot(roty).dot(rotz)


def _project_plane_yz_x(vec):
    x = vec.dot(np.array([0, 1, 0], dtype=np.float32))
    y = vec.dot(np.array([0, 0, 1], dtype=np.float32))
    #return np.array([x, y], dtype=np.float32)  # y flip
    return np.array([x, y], dtype=np.float32)  # y flip

def _project_plane_yz(vec):
    x = vec.dot(np.array([0, 1, 0], dtype=np.float32))
    y = vec.dot(np.array([0, 0, 1], dtype=np.float32))
    #return np.array([x, y], dtype=np.float32)  # y flip
    return np.array([x, -y], dtype=np.float32)  # y flip

def _project_plane_yz_z(vec):
    x = vec.dot(np.array([0, 1, 0], dtype=np.float32))
    y = vec.dot(np.array([0, 0, 1], dtype=np.float32))
    #return np.array([x, y], dtype=np.float32)  # y flip
    return np.array([-x, -y], dtype=np.float32)  # y flip

def draw_pose(img, pose, size=30, idx=0):
    # parallel projection (something wrong?)
    rotmat = _rotation_matrix(-pose[0], -pose[1], -pose[2])
    zvec = np.array([0, 0, 1], np.float32)
    yvec = np.array([0, 1, 0], np.float32)
    xvec = np.array([1, 0, 0], np.float32)
    zvec = _project_plane_yz_z(rotmat.dot(zvec))
    yvec = _project_plane_yz_x(rotmat.dot(yvec))
    xvec = _project_plane_yz(rotmat.dot(xvec))
    #zvec = _project_plane_yz(rotmat.dot(zvec))
    #yvec = _project_plane_yz(rotmat.dot(yvec))
    #xvec = _project_plane_yz(rotmat.dot(xvec))

    
    #cv2.putText(img, str(pose[0]), (10,600), color=(0,0,255), thickness=4, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1)
    #cv2.rectangle(img, (8,img.shape[1] - 170), (70,img.shape[1] - 135), color=(0,0,0), thickness = -1)
    
    #cv2.putText(img, ("x=%d°" % math.degrees(pose[0])), (10,img.shape[1] - 160), color=(255,255,255), thickness=1, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.6)
    #cv2.putText(img, ("y=%d°" % math.degrees(pose[1])), (10,img.shape[1] - 150), color=(255,255,255), thickness=1, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.6)
    #cv2.putText(img, ("z=%d°" % math.degrees(pose[2])), (10,img.shape[1] - 140), color=(255,255,255), thickness=1, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.6)
    # Lower left
    org_pt = ((size + 5) * (2 * idx + 1), img.shape[0] - size - 5)
    _draw_line(img, org_pt, org_pt + zvec * size, (1, 0, 0), 3) #z
    _draw_line(img, org_pt, org_pt + yvec * size, (0, 1, 0), 3) #x
    _draw_line(img, org_pt, org_pt + xvec * size, (0, 0, 1), 3) #y


def draw_gender(img, gender, size=7, idx=0):
    # Upper right
    pt = (img.shape[1] - (size + 5) * (2 * idx + 1), size + 5)
    if gender == 0:
        _draw_circle(img, pt, (1.0, 0.3, 0.3), size, -1)  # male
    elif gender == 1:
        _draw_circle(img, pt, (0.3, 0.3, 1.0), size, -1)  # female


def draw_gender_rect(img, gender, rect):
    if gender == 0:
        _draw_rect(img, rect, (1.0, 0.3, 0.3))  # male
    elif gender == 1:
        _draw_rect(img, rect, (0.3, 0.3, 1.0))  # female
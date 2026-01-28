import cv2
import numpy as np

from .geometry_utils import is_pt_in_poly
from .config import NUMCIRCLE


def anchorCircleDetect(cnt_, val_, corners_, index_aruco_, width_):
    detected_flag_ = 0
    tmp_distance = (cnt_[0][0][0] - val_[0][0][0]) ** 2 + (cnt_[0][0][1] - val_[0][0][1]) ** 2
    if tmp_distance >= width_:
        return detected_flag_
    if is_pt_in_poly(cnt_[0][0], val_[0]) == False:
        return detected_flag_
    rect = cv2.minAreaRect(cnt_)
    rect_center = np.int0(rect[0])
    if is_pt_in_poly(rect_center, val_[0]) == True:
        if is_pt_in_poly(rect_center, corners_[index_aruco_][0]) == False:
            detected_flag_ = 1
        else:
            detected_flag_ = 0
            return detected_flag_
    else:
        detected_flag_ = 0
        return detected_flag_
    corner_distance_range = 0.25
    detected_flag_ = 0
    for i in range(NUMCIRCLE):
        tmp_distance = (rect_center[0] - val_[0][i][0]) ** 2 + (rect_center[1] - val_[0][i][1]) ** 2
        tmp_range = width_ * (corner_distance_range ** 2)
        if tmp_distance <= tmp_range:
            detected_flag_ = 1
            break
    if not detected_flag_:
        return detected_flag_
    box_ = cv2.boxPoints(rect)
    h = np.sqrt((box_[3, 0] - box_[2, 0]) ** 2 + (box_[3, 1] - box_[2, 1]) ** 2)
    w = np.sqrt((box_[2, 0] - box_[1, 0]) ** 2 + (box_[2, 1] - box_[1, 1]) ** 2)
    if h == 0 or w == 0:
        detected_flag_ = 0
        return detected_flag_

    s2 = h * w
    print('s2', s2)
    if s2 < 10. or s2 >= 1000.:
        detected_flag_ = 0
        return detected_flag_

    return detected_flag_


def filter_acp(val_acp_, old_val_acp_, num_pixels=0.):
    err_acp = val_acp_ - old_val_acp_
    [rows, cols] = err_acp.shape
    for w1 in range(rows):
        for w2 in range(cols):
            val_err = err_acp[w1, w2]
            if abs(val_err) <= num_pixels:
                val_acp_[w1, w2] = old_val_acp_[w1, w2]
    old_val_acp_ = np.copy(val_acp_)
    return val_acp_, old_val_acp_

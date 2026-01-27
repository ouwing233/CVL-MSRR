import numpy as np
import cv2


def getRTX(degree):
    radian = degree * np.pi / 180
    Rt = np.array([
        [1, 0, 0],
        [0, np.cos(radian), -np.sin(radian)],
        [0, np.sin(radian), np.cos(radian)]
    ])
    return Rt


def rotateX(vect, degree):
    Rt = getRTX(degree)
    R = cv2.Rodrigues(vect)
    RX = np.dot(R[0], Rt)
    v = cv2.Rodrigues(RX)
    return v[0]


def getRTY(degree):
    radian = degree * np.pi / 180
    Rt = np.array([
        [np.cos(radian), 0, np.sin(radian)],
        [0, 1, 0],
        [-np.sin(radian), 0, np.cos(radian)]
    ])
    return Rt


def rotateY(vect, degree):
    Rt = getRTY(degree)
    R = cv2.Rodrigues(vect)
    RY = np.dot(R[0], Rt)
    v = cv2.Rodrigues(RY)
    return v[0]


def getRTZ(degree):
    radian = degree * np.pi / 180
    Rt = np.array([
        [np.cos(radian), -np.sin(radian), 0],
        [np.sin(radian), np.cos(radian), 0],
        [0, 0, 1]
    ])
    return Rt


def rotateZ(vect, degree):
    Rt = getRTZ(degree)
    R = cv2.Rodrigues(vect)
    RZ = np.dot(R[0], Rt)
    v = cv2.Rodrigues(RZ)
    return v[0]


def rvec2degree(rotate_vector):
    norm = np.linalg.norm(rotate_vector, ord=2)
    rotate_degree = norm * 180 / np.pi
    return rotate_degree


def tranformate2radiusVector(idx_, rvecs_, degree_list_):
    if idx_ < 8:
        tmpr_ = rotateY(vect=rvecs_, degree=degree_list_[idx_])
    elif idx_ < 16:
        tmpr_ = rotateX(vect=rvecs_, degree=degree_list_[idx_ - 7])
    elif idx_ < 24:
        tmpr_ = rotateX(vect=rvecs_, degree=degree_list_[2])
        tmpr_ = rotateY(vect=tmpr_, degree=-degree_list_[idx_ - 15])
    return tmpr_


def calc_transrot(idx_, degree_list_):
    if idx_ < 6:
        transRotation = getRTY(degree=degree_list_[3 - idx_])
    elif idx_ < 12:
        transRotation = getRTX(degree=degree_list_[3 - (idx_ - 6)])
    return transRotation


def adjustEulerBias(euler_list):
    [rows, cols] = euler_list.shape
    for row in range(rows):
        for col in range(cols):
            val_bias = abs(euler_list[row, col])
            if val_bias >= 180.0:
                val_bias = 360.0 - val_bias
            euler_list[row, col] = val_bias
    return euler_list


def adjustEuler2Positive(euler_array):
    [rows, cols] = euler_array.shape
    for row in range(rows):
        for col in range(cols):
            val_euler = euler_array[row, col]
            if val_euler < 0:
                val_euler = val_euler + 360.
            euler_array[row, col] = val_euler
    return euler_array


def adjustEuler2normal(euler_array):
    [rows, cols] = euler_array.shape
    for row in range(rows):
        for col in range(cols):
            val_euler = euler_array[row, col]
            if val_euler > 180.:
                val_euler = val_euler - 360.
            euler_array[row, col] = val_euler
    return euler_array

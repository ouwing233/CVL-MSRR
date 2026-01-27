import numpy as np


def calc_border(value):
    left_border = value[0, 0][0]
    right_border = value[0, 0][0]
    up_border = value[0, 0][1]
    down_border = value[0, 0][1]
    for val_j in range(0, 4):
        if left_border > value[0, val_j][0]:
            left_border = value[0, val_j][0]
        if right_border < value[0, val_j][0]:
            right_border = value[0, val_j][0]
        if up_border > value[0, val_j][1]:
            up_border = value[0, val_j][1]
        if down_border < value[0, val_j][1]:
            down_border = value[0, val_j][1]
    border_array = np.array(([left_border, right_border, up_border, down_border]), dtype=int)
    return border_array


def calc_all_aruco_border(value):
    border_list = []
    for i, val in enumerate(value):
        tmp_border = calc_border(val)
        border_list.append(tmp_border)
    return border_list


def Node_Angle(a, b, c):
    Vbx = b[0] - a[0]
    Vby = b[1] - a[1]
    Vcx = c[0] - a[0]
    Vcy = c[1] - a[1]
    angle_line = (Vbx * Vcx + Vby * Vcy) / np.sqrt((Vbx * Vbx + Vby * Vby) * (Vcx * Vcx + Vcy * Vcy) + 1e-10)
    angle = np.arccos(angle_line) * 180.0 / np.pi
    return round(angle, 3)


def is_pt_in_poly(pt, poly):
    nvert = len(poly)
    vertx = []
    verty = []
    testx = pt[0]
    testy = pt[1]
    for item in poly:
        vertx.append(item[0])
        verty.append(item[1])

    j = nvert - 1
    res = False
    for i in range(nvert):
        if (verty[j] - verty[i]) == 0:
            j = i
            continue
        x = (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i]
        if ((verty[i] > testy) != (verty[j] > testy)) and (testx < x):
            res = not res
        j = i
    return res


def sort_4_Vector(tmp_point_, cornerP):
    array_pt = np.array(tmp_point_)
    final_point_ = []
    v1 = array_pt[1] - array_pt[0]
    v2 = array_pt[2] - array_pt[0]
    v3 = array_pt[3] - array_pt[0]
    v1_2 = np.cross(v1, v2)
    v1_3 = np.cross(v1, v3)
    v2_3 = np.cross(v2, v3)
    
    if (v1_2 > 0) and (v1_3 > 0) and (v2_3 > 0):
        final_point_.append(array_pt[0])
        final_point_.append(array_pt[1])
        final_point_.append(array_pt[2])
        final_point_.append(array_pt[3])
    elif (v1_2 > 0) and (v1_3 > 0) and (v2_3 < 0):
        final_point_.append(array_pt[0])
        final_point_.append(array_pt[1])
        final_point_.append(array_pt[3])
        final_point_.append(array_pt[2])
    elif (v1_2 > 0) and (v1_3 < 0) and (v2_3 < 0):
        final_point_.append(array_pt[0])
        final_point_.append(array_pt[3])
        final_point_.append(array_pt[1])
        final_point_.append(array_pt[2])
    elif (v1_2 < 0) and (v1_3 < 0) and (v2_3 > 0):
        final_point_.append(array_pt[0])
        final_point_.append(array_pt[2])
        final_point_.append(array_pt[3])
        final_point_.append(array_pt[1])
    elif (v1_2 < 0) and (v1_3 < 0) and (v2_3 < 0):
        final_point_.append(array_pt[0])
        final_point_.append(array_pt[3])
        final_point_.append(array_pt[2])
        final_point_.append(array_pt[1])
    elif (v1_2 < 0) and (v1_3 > 0) and (v2_3 > 0):
        final_point_.append(array_pt[0])
        final_point_.append(array_pt[2])
        final_point_.append(array_pt[1])
        final_point_.append(array_pt[3])
    else:
        final_point_.append(array_pt[0])
        final_point_.append(array_pt[1])
        final_point_.append(array_pt[2])
        final_point_.append(array_pt[3])

    array_pt = np.array(final_point_)
    a1 = Node_Angle(array_pt[0], array_pt[1], array_pt[3])
    a2 = Node_Angle(array_pt[1], array_pt[2], array_pt[0])
    a3 = Node_Angle(array_pt[2], array_pt[3], array_pt[1])
    a4 = Node_Angle(array_pt[3], array_pt[0], array_pt[2])
    anglelt = [a1, a2, a3, a4]
    
    d1 = np.linalg.norm((cornerP - final_point_[0]), ord=2)
    d2 = np.linalg.norm((cornerP - final_point_[1]), ord=2)
    d3 = np.linalg.norm((cornerP - final_point_[2]), ord=2)
    d4 = np.linalg.norm((cornerP - final_point_[3]), ord=2)
    dlist = [d1, d2, d3, d4]
    first_p = dlist.index(min(dlist))
    final_point_.extend(final_point_[0:first_p])
    final_point_ = final_point_[first_p:]
    return final_point_

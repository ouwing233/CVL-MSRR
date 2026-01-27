import cv2
import cv2.aruco as aruco
import time
import numpy as np
import threading
import math
from math import sin, cos, pi
from maix import display, camera, image
from PIL import Image 
from itertools import *
from scipy.spatial.transform import Rotation

radius_vec = np.array([0, 0, 32.5], dtype=float)
radius_vec2 = np.array([0, 0, 30.0], dtype=float)
width_gd = 18.
width_pix_g1 = 338
width_pix_g2 = 618 
distance_gd = 150.
focal_length = distance_gd * width_pix_g1 / width_gd

def getRTX(degree):
    radian = degree * np.pi / 180
    Rt = np.array([[1, 0, 0],
                    [0, np.cos(radian), -np.sin(radian)],
                    [0, np.sin(radian), np.cos(radian)]])
    return Rt

def rotateX(vect, degree):
    Rt = getRTX(degree)
    R = cv2.Rodrigues(vect)
    RX = np.dot(R[0], Rt)
    v = cv2.Rodrigues(RX)
    return v[0]

def getRTY(degree):
    radian = degree * np.pi / 180
    Rt = np.array([[np.cos(radian), 0, np.sin(radian)],
                    [0, 1, 0],
                    [-np.sin(radian), 0, np.cos(radian)]])
    return Rt

def rotateY(vect, degree):
    Rt = getRTY(degree)
    R = cv2.Rodrigues(vect)
    RY = np.dot(R[0], Rt)
    v = cv2.Rodrigues(RY)
    return v[0]

def getRTZ(degree):
    radian = degree * np.pi / 180
    Rt = np.array([[np.cos(radian), -np.sin(radian), 0],
                    [np.sin(radian), np.cos(radian), 0],
                    [0, 0, 1]])
    return Rt

def rotateZ(vect, degree):
    Rt = getRTZ(degree)
    R = cv2.Rodrigues(vect)
    RZ = np.dot(R[0], Rt)
    v = cv2.Rodrigues(RZ)
    return v[0]

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
    ab = float(np.sqrt(pow((a[0] - b[0]), 2) + pow((a[1] - b[1]), 2)))
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

def rvec2degree(rotate_vector):
    norm = np.linalg.norm(rotate_vector, ord=2)
    rotate_degree = norm * 180 / np.pi
    return rotate_degree

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

def tranformate2radiusVector(idx_, rvecs_, degree_list_):
    if (idx_ < 8):
        tmpr_ = rotateY(vect=rvecs_, degree=degree_list_[idx_])
    elif (idx_ < 16):
        tmpr_ = rotateX(vect=rvecs_, degree=degree_list_[idx_ - 7])
    elif (idx_ < 24):
        tmpr_ = rotateX(vect=rvecs_, degree=degree_list_[2])
        tmpr_ = rotateY(vect=tmpr_, degree=-degree_list_[idx_ - 15])
    return tmpr_

def calc_transrot(idx_, degree_list_):
    if (idx_ < 6):
        transRotation = getRTY(degree=degree_list_[3 - idx_])
    elif (idx_ < 12):
        transRotation = getRTX(degree=degree_list_[3 - (idx_ - 6)])
    return transRotation

def adjustEulerBias(euler_list):
    [rows, cols] = euler_list.shape
    for row in range(rows):
        for col in range(cols):
            val_bias = abs(euler_list[row,col])
            if val_bias >= 180.0:
                val_bias = 360.0 - val_bias
            euler_list[row,col] = val_bias
    return euler_list

def adjustEuler2Positive(euler_array):
    [rows, cols] = euler_array.shape
    for row in range(rows):
        for col in range(cols):
            val_euler = euler_array[row,col]
            if val_euler < 0:
                val_euler = val_euler + 360.
            euler_array[row,col] = val_euler
    return euler_array

def adjustEuler2normal(euler_array):
    [rows, cols] = euler_array.shape
    for row in range(rows):
        for col in range(cols):
            val_euler = euler_array[row,col]
            if val_euler > 180.:
                val_euler = val_euler-360.
            euler_array[row,col] = val_euler
    return euler_array

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
    d1 = np.linalg.norm((cornerP - final_point_[0]),ord=2)
    d2 = np.linalg.norm((cornerP - final_point_[1]),ord=2)
    d3 = np.linalg.norm((cornerP - final_point_[2]),ord=2)
    d4 = np.linalg.norm((cornerP - final_point_[3]),ord=2)
    dlist = [d1, d2, d3, d4]
    first_p = dlist.index(min(dlist))
    final_point_.extend(final_point_[0:first_p])
    final_point_ = final_point_[first_p:]
    return final_point_

k_outline_range = 0.8
savePath = './failedPicture/'
CAMNO = 1
FIGWIDTH = 1280
FIGHEIGHT = 720
NUMCIRCLE = 4
NUMID = 256
WINSIZE = 4
FILTER_THRESHOLD = 0.0
SERVERIP = '192.168.1.113'
gt_3d_points = np.array(([-25, 25, 0], [25, 25, 0], [25, -25, 0], [-25, -25, 0]), dtype=np.double)
marker_num = 6
ept = - 360. / marker_num
degree_list = {0: 0, 1: 1 * ept, 2: 2 * ept, 3: 3 * ept, -1: -1 * ept, -2: -2 * ept, -3: -3 * ept
                }
decay = 0.8

useExtrinsicGuess = True
lk_params = dict( winSize  = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))    

import socket
import time
send_PORT_default = 8888
bind_PORT = 9997

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
s.bind(('', bind_PORT))
s.settimeout(3)
print('Bind UDP on ', bind_PORT, '...')

def UDP_receive():
    global s
    try:
        data, addr = s.recvfrom(1024)
        data = data.decode("utf-8","ignore")
        message = 'Received from (%s:%s),%s' % (addr[0],addr[1], data)
        print(message)
    except Exception as e:
        print(e)

def detector():
    global idList1, data_cam1, s
    data_position_rvec = [None]*NUMID
    for i in range(len(data_position_rvec)):
        data_position_rvec[i] = [[], [], [], 0]
    data_cam1 = data_position_rvec
    idList = []
    idList1 = idList
    ids = ()
    ball_center_list = []
    ball_center = np.array([0., 0., 0.])
    old_bc = np.array([0., 0., 0.])
    old_rvec = np.zeros([3,1],dtype=float)
    old_val_acp = np.zeros((NUMID,NUMCIRCLE,2),dtype=float)
    old_euler = np.zeros((1,3),dtype=float)
    tmpr = [[0], [0], [0]]
    oldtmpr = [[0], [0], [0]]
    old_rotateDegree = 0
    old_corners = ()
    effective_sign = []
    anchorCnt = []
    anchorPoint = []
    indexList = []
    ball_center_list = []
    rvec_list = []
    time_last = time.time()
    frame = np.array([0.])
    pose = {"x": 0., "y": 0., "z": 0.}
    ncount = 0
    flag_track = False
    tracks = []
    idx_track = -1
    FrameCnt = 0
    fcnt = 0
    
    cameraMatrix = np.array([[1226.94278978,    0.,          624.40778315],
                             [   0.,         1226.17471794,  372.19401808],
                             [   0.,            0.,            1.,        ]], dtype=float)
    distCoeffs = np.array([ 0.1138495,  -0.8523845,  -0.00100567, -0.00059563,  1.33241712], dtype=float)
    fx = cameraMatrix[0][0]
    cx = cameraMatrix[0][2]
    fy = cameraMatrix[1][1]
    cy = cameraMatrix[1][2]
    cap = cv2.VideoCapture(0)
    cap.set(3,FIGWIDTH)
    cap.set(4,FIGHEIGHT)

    while True:
        ttoal = time.time()
        suc, raw_frame = cap.read()
        t1 = time.time()
        frame = raw_frame
        vis = frame.copy()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        frame = frame.astype(np.uint8)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flag_track = False
        if (len(idList)>0) and (len(tracks) == NUMCIRCLE*len(idList)) :
            t1 = time.time()
            length_track = len(tracks)
            track_len = 10
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    new_tracks.append([(0,0)])
                    continue
                tr.append((x, y))
                if len(tr) > track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
            tracks = new_tracks
            cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
            prev_gray = frame_gray
            flag_track = True
            content_total = ''
            for idx_track in range(len(idList)):
                val_ = []
                for val_idx in range(idx_track*NUMCIRCLE, (idx_track+1)*NUMCIRCLE):
                    if good[val_idx]:
                        val_.append(tracks[val_idx][-1])
                    else:
                        break
                if len(val_) < NUMCIRCLE:
                    flag_track = False
                    continue
                val_ = np.array(val_,dtype=float)
                for k in range(0, len(val_)):
                    cv2.putText(vis, str(k), (int(val_[k][0]), int(val_[k][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (255, 0, 0), 1)
                transfom_set = cv2.solvePnPGeneric(gt_3d_points, val_, cameraMatrix,
                                                                                    distCoeffs,
                                                                                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                                                                                )
                err1 = transfom_set[3][0]
                err2 = transfom_set[3][1]
                rotation_vector = transfom_set[1][0]
                translation_vector = transfom_set[2][0]

                rvecs = rotation_vector
                tvecs = translation_vector
                tmpr = rotation_vector.reshape(1,3)
                print("track RP",int(idList[idx_track]),':', err1)
                tmplength = len(indexList)
                meanrvec = np.array([[0.], [0.], [0.]])
                meanrvec = rvecs

                time_this = time.time()
                if 0 != data_position_rvec[int(idList[idx_track])][3]:
                    time_last = data_position_rvec[int(idList[idx_track])][3]
                else:
                    time_last = np.copy(time_this)
                rotateDegree = rvec2degree(meanrvec)
                if len(data_position_rvec[int(idList[idx_track])][1])>0:
                    old_rotateDegree = rvec2degree(data_position_rvec[int(idList[idx_track])][1])
                    oldtmpr = np.copy(data_position_rvec[int(idList[idx_track])][1])
                else:
                    old_rotateDegree = 0
                degree_bias = rotateDegree - old_rotateDegree

                if 0 <= int(idList[idx_track]) and int(idList[idx_track]) < 24:
                    aruco.drawAxis(vis, cameraMatrix, distCoeffs, meanrvec,
                                   np.array([-0.2409056, 0.10717732, 0.76153129]), 0.1)
                if 24 <= int(idList[idx_track]) and int(idList[idx_track]) < 48:
                    aruco.drawAxis(vis, cameraMatrix, distCoeffs, meanrvec,
                                   np.array([-0.3409056, 0.20717732, 0.76153129]), 0.1)

                ball_center = np.array([0., 0., 0.])
                left_up = val_[0]
                right_up = val_[1]
                right_down = val_[2]
                center_x = left_up[0]
                center_y = left_up[1]
                temp_center = np.array([center_x, center_y], dtype=int)
                d = distance_gd * tvecs[2] / width_pix_g1
                pz = d
                px = (center_x - cx) * pz / fx
                py = (center_y - cy) * pz / fy

                ball_center[0] = px
                ball_center[1] = py
                ball_center[2] = pz

                rvec2rmat = cv2.Rodrigues(meanrvec)
                pr = np.dot(radius_vec, rvec2rmat[0])

                old_bc = np.copy(ball_center)

                if len(corners) == 0:
                    ball_center = old_bc
                ball_center_list.append(ball_center)
                rvec_list.append(meanrvec)
                data_position_rvec[int(idList[idx_track])][0] = np.copy(ball_center)
                data_position_rvec[int(idList[idx_track])][1] = np.copy(meanrvec)
                data_position_rvec[int(idList[idx_track])][2] = np.copy(tvecs)
                data_position_rvec[int(idList[idx_track])][3] = np.copy(time_this)
                FrameCnt += 1
                content2 = '\nCamNo:{0}\nFrame:{1}\nidx:{2}\nbc:{3}\nrvec:{4}\ntvec:{5}\ntime:{6}'.format(str(CAMNO), str(FrameCnt), str(idList[idx_track]), repr(ball_center), \
                    repr(meanrvec.reshape(1,3)), repr(tvecs.reshape(1,3)), str(time_this))
                print(content2)
                content_total = content_total + content2
                cv2.putText(vis, 'tracking id='+str(int(idList[idx_track]))+' time cost='+str(time.time()-ttoal)+'flag_track='+str(flag_track), \
                                (int(1), int(FIGHEIGHT-1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 127, 0), 2)
            s.sendto(content_total.encode("utf-8"), (SERVERIP, bind_PORT))
        if flag_track == True:
            vis = cv2.resize(vis, (240, 240))
            vis = cv2.flip(vis, -1)
            img = image.load(vis[..., ::-1].tobytes(), (240, 240))
            display.show(img, remote_show=False)
            continue
        effective_sign.clear()
        anchorCnt.clear()
        anchorPoint.clear()
        indexList.clear()
        idList.clear()
        ball_center_list.clear()
        rvec_list.clear()
        ttoal = time.time()

        if len(corners) <= 0:
            print('can not find any aruco')
            vis = vis.astype(np.uint8)
            cv2.putText(vis, str(CAMNO), (int(FIGWIDTH/2), int(FIGHEIGHT/10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (255, 0, 0), 2)

            vis = cv2.resize(vis, (240, 240))
            vis = cv2.flip(vis, -1)
            img = image.load(vis[..., ::-1].tobytes(), (240, 240))
            display.show(img, remote_show=False)
            continue
        outline_border = []
        for id_arcuo, cnt in enumerate(corners):
            vec1 = (cnt[0][0] - cnt[0][2]) * k_outline_range
            vec2 = (cnt[0][1] - cnt[0][3]) * k_outline_range
            p_leftup = cnt[0][0] + vec1
            p_rightdown = cnt[0][2] - vec1
            p_rightup = cnt[0][1] + vec2
            p_leftdown = cnt[0][3] - vec2
            tmp_ot = [p_leftup, p_rightup, p_rightdown, p_leftdown]
            for j in range(len(tmp_ot)):
                if tmp_ot[j][0] < 0:
                    tmp_ot[j][0] = 0
                if tmp_ot[j][0] >= FIGWIDTH:
                    tmp_ot[j][0] = FIGWIDTH - 1
                if tmp_ot[j][1] < 0:
                    tmp_ot[j][1] = 0
                if tmp_ot[j][1] >= FIGWIDTH:
                    tmp_ot[j][1] = FIGWIDTH - 1
            tmp_ot = np.array(tmp_ot,dtype=np.float32)
            tmp_ot = tmp_ot.reshape(1,4,2)
            outline_border.append(tmp_ot)
        outline_border = tuple(outline_border)
        rejectedImgPoints = outline_border
        if flag_track == False:
            tracks = []
            for i, val in enumerate(rejectedImgPoints):
                border_ary = calc_border(val)
                tmp_cnt = []
                num_circle = 0
                effective_flag = 0
                num_effective_aruco = 0
                index_aruco = 0
                wdt = border_ary[1] - border_ary[0]
                lgt = border_ary[3] - border_ary[2]
                s1 = wdt * lgt
                tmp_frame = frame[border_ary[2]:border_ary[3], border_ary[0]:border_ary[1], :]
                hsv=cv2.cvtColor(tmp_frame,cv2.COLOR_BGR2HSV)
                lower=np.array([0,0,0])
                upper=np.array([180,255,75])
                mask=cv2.inRange(hsv,lower,upper)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(tmp_frame, contours, -1, (0, 255, 255), 1)
                offset = np.array([border_ary[0],border_ary[2]],int)
                offset = offset.reshape(1,1,2)

                t1 = time.time()
                if len(corners) <= 0:
                    break
                else:
                    for id_arcuo, cnt in enumerate(corners):
                        ar_detected_flag = 0
                        for point_e in cnt[0]:
                            if is_pt_in_poly(point_e, val[0]) == True:
                                ar_detected_flag = 1
                            else:
                                ar_detected_flag = 0
                                print('this aruco is not in it')
                                break
                        if ar_detected_flag == 1:
                            num_effective_aruco += 1
                            if num_effective_aruco == 1:
                                index_aruco = id_arcuo
                            elif num_effective_aruco > 1:
                                ar_detected_flag = 0
                                num_effective_aruco = 0
                                print('too much aruco in a marker')
                                break
                print('KUANG detect', time.time()-t1)
                t1 = time.time()
                if num_effective_aruco != 1:
                    continue
                else:
                    width_marker = (val[0][2][0] - val[0][0][0]) ** 2 + (val[0][2][1] - val[0][0][1]) ** 2
                    for k, cnt in enumerate(contours):
                        cnt = cnt + offset
                        detected_flag = anchorCircleDetect(cnt, val, corners, index_aruco, width_marker)

                        if detected_flag == 1:
                            tmp_cnt.append(cnt)
                            cv2.drawContours(vis, contours, k, (0, 0, 255), 1)
                            num_circle += 1
                            tmpCNT = cnt - offset
                            print('id=',num_circle)
                            cv2.putText(vis, str(num_circle), (int(tmpCNT[0][0][0]), int(tmpCNT[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (255, 0, 0), 1)
                if num_circle>=NUMCIRCLE:
                    good_circle = []
                    final_point = []
                    for tmp_cnt_unit in combinations(tmp_cnt,4):
                        unit_list = list(tmp_cnt_unit)
                        tmp_point = []
                        for k2, cnt2 in enumerate(unit_list[0:NUMCIRCLE]):
                            rect = cv2.minAreaRect(cnt2)
                            box_ = cv2.boxPoints(rect)
                            h = abs(box_[3, 1] - box_[1, 1])
                            w = abs(box_[3, 0] - box_[1, 0])
                            if (h > 500 or w > 500):
                                continue
                            box = cv2.boxPoints(rect)
                            box = np.int0(box)
                            angle = rect[2]
                            tmp_point.append(list(rect[0]))
                        cornerP = np.array(corners[int(index_aruco)][0, 0],dtype=int)
                        final_point = sort_4_Vector(tmp_point, cornerP)
                        flag_aruco_in = True
                        for k in range(0, 4):
                            if is_pt_in_poly(corners[int(index_aruco)][0, k], final_point) == False:
                                print('!!!!it is not have any aruco there!!!!!')
                                flag_aruco_in = False
                                break
                        if flag_aruco_in == False:
                            continue
                        else:
                            good_point = final_point
                            break
                
                    if flag_aruco_in == True:
                        effective_flag = 1
                        effective_sign.append(val)
                        anchorPoint.append(np.array(good_point))
                        indexList.append(index_aruco)
                        idList.append(int(ids[index_aruco]))
                        prev_gray = frame_gray
                        print('successfully got!')
                    else:
                        effective_flag = 0
                        print('failed got!')
                        fcnt += 1
                print('anchor detect', time.time()-t1)
        tuple(effective_sign)
        tuple(anchorCnt)
        aruco.drawDetectedMarkers(vis, corners, ids)
        aruco.drawDetectedMarkers(vis, rejectedImgPoints, )
        t1 = time.time()
        if len(effective_sign) > 0:
            t1 = time.time()
            flag_draw1 = False
            flag_draw2 = False
            FrameCnt += 1
            content_total = ''
            for index_val, val_acp in enumerate(anchorPoint):
                idx = int(ids[indexList[index_val]])

                [val_acp, old_val_acp[idx]] = filter_acp(val_acp, old_val_acp[idx], num_pixels=FILTER_THRESHOLD)
                val_acp.dtype = np.float
                rvecs = data_position_rvec[idx][1]
                tvecs = data_position_rvec[idx][2]
                rotation_vector_init = data_position_rvec[idx][1]
                translation_vector_init = data_position_rvec[idx][2]
                r2 = np.zeros((3,1),float)
                useExtrinsicGuess = True
                for k in range(0, len(val_acp)):
                    cv2.putText(vis, str(k), (int(val_acp[k][0]), int(val_acp[k][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (255, 0, 0), 1)
                    cv2.circle(vis, (int(val_acp[k][0]), int(val_acp[k][1])), 1, (0, 0, 255), 1)
                for x, y in np.float32(val_acp).reshape(-1, 2):
                    tracks.append([(x, y)])
                if len(rvecs) == 0 or len(tvecs) == 0:
                    (success, rotation_vector_init, translation_vector_init) = cv2.solvePnP(gt_3d_points, val_acp, cameraMatrix,
                                                                                distCoeffs, 
                                                                                flags=cv2.SOLVEPNP_ITERATIVE)
                    rotation_vector = rotation_vector_init
                    translation_vector = translation_vector_init
                    err1 = 0
                    print('init')
                else:
                    transfom_set = cv2.solvePnPGeneric(gt_3d_points, val_acp, cameraMatrix,
                                                                                distCoeffs,
                                                                                flags=cv2.SOLVEPNP_IPPE_SQUARE
                                                                            )
                    if transfom_set[0] != 0:
                        rotation_vector = transfom_set[1][0]
                        r2 = transfom_set[1][1]
                        translation_vector = transfom_set[2][0]
                        t2 = transfom_set[2][1]
                        err1 = transfom_set[3][0]
                        err2 = transfom_set[3][1]
                    else:
                        print('calculate failed')
                        continue
                
                    if (err1 >= 0.1 and err2 / err1 > 2.0) or (err2 / err1 <= 2.0):
                        print('recalculate!')
                        transfom_set = cv2.solvePnPGeneric(gt_3d_points, val_acp, cameraMatrix,
                                                                                distCoeffs,useExtrinsicGuess=useExtrinsicGuess,
                                                                                flags=cv2.SOLVEPNP_ITERATIVE,rvec=rotation_vector_init,tvec=translation_vector_init
                                                                            )
                        rotation_vector = transfom_set[1][0]
                        translation_vector = transfom_set[2][0]
                        err1 = transfom_set[3][0]
                rvecs = rotation_vector
                tvecs = translation_vector

                print("RP",idx,':', err1)

                maxlen = 1
                tmplength = len(indexList)
                meanrvec = np.array([[0.], [0.], [0.]])
                meanrvec = rvecs

                time_this = time.time()
                if 0 != data_position_rvec[idx][3]:
                    time_last = data_position_rvec[idx][3]
                else:
                    time_last = np.copy(time_this)


                if 0 <= idx and idx < 24 and flag_draw1 == False:
                    flag_draw1 = True
                if 24 <= idx and idx < 48 and flag_draw2 == False:
                    flag_draw2 = True

                ball_center = np.array([0., 0., 0.])
                left_up = val_acp[0]
                right_up = val_acp[1]
                left_down = val_acp[2]
                center_x = left_up[0]
                center_y = left_up[1]
                temp_center = np.array([center_x, center_y], dtype=int)
                d = distance_gd * tvecs[2] / width_pix_g1
                pz = d
                px = (center_x - cx) * pz / fx
                py = (center_y - cy) * pz / fy

                ball_center[0] = px
                ball_center[1] = py
                ball_center[2] = pz

                rvec2rmat = cv2.Rodrigues(meanrvec)
                pr = np.dot(radius_vec, rvec2rmat[0])

                old_bc = np.copy(ball_center)

                if len(corners) == 0:
                    ball_center = old_bc
                ball_center_list.append(ball_center)
                rvec_list.append(meanrvec)
                data_position_rvec[idx][0] = np.copy(ball_center)
                data_position_rvec[idx][1] = np.copy(meanrvec)
                data_position_rvec[idx][2] = np.copy(tvecs)
                data_position_rvec[idx][3] = np.copy(time_this)
                content2 = '\nCamNo:{0}\nFrame:{1}\nidx:{2}\nbc:{3}\nrvec:{4}\ntvec:{5}\ntime:{6}'.format(str(CAMNO), str(FrameCnt), str(idx), repr(ball_center), \
                    repr(meanrvec.reshape(1,3)), repr(tvecs.reshape(1,3)), str(time_this))
                print(content2)
                content_total = content_total + content2
            
            s.sendto(content_total.encode("utf-8"), (SERVERIP, bind_PORT))

        vis = vis.astype(np.uint8)
        cv2.putText(vis, str(CAMNO), (int(FIGWIDTH/2), int(FIGHEIGHT/10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (255, 0, 0), 2)
        vis = cv2.resize(vis, (240, 240))
        vis = cv2.flip(vis, -1)
        img = image.load(vis[..., ::-1].tobytes(), (240, 240))
        display.show(img, remote_show=False)
        print('total',time.time()-ttoal)




if __name__ == '__main__':
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    dictLen = 250
    parameters = aruco.DetectorParameters_create()
    np.set_printoptions(suppress=True)
    t1 = threading.Thread(target=detector)
    tlist = [t1, ]
    try:
        for k in tlist:
            k.setDaemon(True)
        for k in tlist:
            k.start()
    except Exception as e:
        print(e)

    t1.join()

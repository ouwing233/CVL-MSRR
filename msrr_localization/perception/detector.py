import cv2
import cv2.aruco as aruco
import time
import numpy as np
from itertools import combinations
from maix import display, camera, image

from .config import (
    CAMNO, FIGWIDTH, FIGHEIGHT, NUMCIRCLE, NUMID, FILTER_THRESHOLD,
    SERVERIP, bind_PORT, gt_3d_points, radius_vec, distance_gd, 
    width_pix_g1, k_outline_range, lk_params
)
from .geometry_utils import calc_border, is_pt_in_poly, sort_4_Vector
from .aruco_utils import anchorCircleDetect, filter_acp
from .rotation_utils import rvec2degree
from .network_utils import create_udp_socket


class ArucoDetector:
    
    def __init__(self):
        self.cameraMatrix = np.array([
            [1226.94278978, 0., 624.40778315],
            [0., 1226.17471794, 372.19401808],
            [0., 0., 1.]
        ], dtype=float)
        
        self.distCoeffs = np.array([0.1138495, -0.8523845, -0.00100567, -0.00059563, 1.33241712], dtype=float)
        
        self.fx = self.cameraMatrix[0][0]
        self.cx = self.cameraMatrix[0][2]
        self.fy = self.cameraMatrix[1][1]
        self.cy = self.cameraMatrix[1][2]
        
        self.data_position_rvec = [None] * NUMID
        for i in range(len(self.data_position_rvec)):
            self.data_position_rvec[i] = [[], [], [], 0]
        
        self.idList = []
        self.old_val_acp = np.zeros((NUMID, NUMCIRCLE, 2), dtype=float)
        
        self.tracks = []
        self.flag_track = False
        self.prev_gray = None
        
        self.FrameCnt = 0
        self.fcnt = 0
        
        self.socket = create_udp_socket()
        
    def _process_optical_flow(self, frame_gray, vis):
        if (len(self.idList) > 0) and (len(self.tracks) == NUMCIRCLE * len(self.idList)):
            track_len = 10
            img0, img1 = self.prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    new_tracks.append([(0, 0)])
                    continue
                tr.append((x, y))
                if len(tr) > track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
            
            self.tracks = new_tracks
            cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
            self.prev_gray = frame_gray
            return True, good
        return False, None

    def _process_tracking(self, vis, good, ttoal):
        content_total = ''
        flag_track = True
        
        for idx_track in range(len(self.idList)):
            val_ = []
            for val_idx in range(idx_track * NUMCIRCLE, (idx_track + 1) * NUMCIRCLE):
                if good[val_idx]:
                    val_.append(self.tracks[val_idx][-1])
                else:
                    break
            
            if len(val_) < NUMCIRCLE:
                flag_track = False
                continue
            
            val_ = np.array(val_, dtype=float)
            for k in range(0, len(val_)):
                cv2.putText(vis, str(k), (int(val_[k][0]), int(val_[k][1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1)
            
            transfom_set = cv2.solvePnPGeneric(gt_3d_points, val_, self.cameraMatrix,
                                                self.distCoeffs,
                                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            err1 = transfom_set[3][0]
            rotation_vector = transfom_set[1][0]
            translation_vector = transfom_set[2][0]

            rvecs = rotation_vector
            tvecs = translation_vector
            meanrvec = rvecs

            print("track RP", int(self.idList[idx_track]), ':', err1)

            time_this = time.time()
            idx = int(self.idList[idx_track])
            
            if 0 != self.data_position_rvec[idx][3]:
                time_last = self.data_position_rvec[idx][3]
            else:
                time_last = np.copy(time_this)

            if 0 <= idx < 24:
                aruco.drawAxis(vis, self.cameraMatrix, self.distCoeffs, meanrvec,
                              np.array([-0.2409056, 0.10717732, 0.76153129]), 0.1)
            if 24 <= idx < 48:
                aruco.drawAxis(vis, self.cameraMatrix, self.distCoeffs, meanrvec,
                              np.array([-0.3409056, 0.20717732, 0.76153129]), 0.1)

            ball_center = self._calculate_ball_center(val_, tvecs)
            
            rvec2rmat = cv2.Rodrigues(meanrvec)
            pr = np.dot(radius_vec, rvec2rmat[0])

            self.data_position_rvec[idx][0] = np.copy(ball_center)
            self.data_position_rvec[idx][1] = np.copy(meanrvec)
            self.data_position_rvec[idx][2] = np.copy(tvecs)
            self.data_position_rvec[idx][3] = np.copy(time_this)
            self.FrameCnt += 1
            
            content2 = '\nCamNo:{0}\nFrame:{1}\nidx:{2}\nbc:{3}\nrvec:{4}\ntvec:{5}\ntime:{6}'.format(
                str(CAMNO), str(self.FrameCnt), str(idx), repr(ball_center),
                repr(meanrvec.reshape(1, 3)), repr(tvecs.reshape(1, 3)), str(time_this))
            print(content2)
            content_total = content_total + content2
            cv2.putText(vis, 'tracking id=' + str(idx) + ' time cost=' + str(time.time() - ttoal) + 
                       'flag_track=' + str(flag_track),
                       (int(1), int(FIGHEIGHT - 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 127, 0), 2)
        
        self.socket.sendto(content_total.encode("utf-8"), (SERVERIP, bind_PORT))
        return flag_track

    def _calculate_ball_center(self, val_, tvecs):
        ball_center = np.array([0., 0., 0.])
        left_up = val_[0]
        center_x = left_up[0]
        center_y = left_up[1]
        d = distance_gd * tvecs[2] / width_pix_g1
        pz = d
        px = (center_x - self.cx) * pz / self.fx
        py = (center_y - self.cy) * pz / self.fy

        ball_center[0] = px
        ball_center[1] = py
        ball_center[2] = pz
        return ball_center

    def _calculate_outline_border(self, corners):
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
            
            tmp_ot = np.array(tmp_ot, dtype=np.float32)
            tmp_ot = tmp_ot.reshape(1, 4, 2)
            outline_border.append(tmp_ot)
        return tuple(outline_border)

    def _detect_markers(self, frame, frame_gray, corners, ids, vis):
        self.tracks = []
        effective_sign = []
        anchorPoint = []
        indexList = []
        
        rejectedImgPoints = self._calculate_outline_border(corners)
        
        for i, val in enumerate(rejectedImgPoints):
            border_ary = calc_border(val)
            tmp_cnt = []
            num_circle = 0
            num_effective_aruco = 0
            index_aruco = 0
            
            tmp_frame = frame[border_ary[2]:border_ary[3], border_ary[0]:border_ary[1], :]
            hsv = cv2.cvtColor(tmp_frame, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 0, 0])
            upper = np.array([180, 255, 75])
            mask = cv2.inRange(hsv, lower, upper)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(tmp_frame, contours, -1, (0, 255, 255), 1)
            offset = np.array([border_ary[0], border_ary[2]], int)
            offset = offset.reshape(1, 1, 2)

            if len(corners) <= 0:
                break
            
            for id_arcuo, cnt in enumerate(corners):
                ar_detected_flag = 0
                for point_e in cnt[0]:
                    if is_pt_in_poly(point_e, val[0]):
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

            if num_effective_aruco != 1:
                continue
            
            width_marker = (val[0][2][0] - val[0][0][0]) ** 2 + (val[0][2][1] - val[0][0][1]) ** 2
            for k, cnt in enumerate(contours):
                cnt = cnt + offset
                detected_flag = anchorCircleDetect(cnt, val, corners, index_aruco, width_marker)

                if detected_flag == 1:
                    tmp_cnt.append(cnt)
                    cv2.drawContours(vis, contours, k, (0, 0, 255), 1)
                    num_circle += 1
                    tmpCNT = cnt - offset
                    print('id=', num_circle)
                    cv2.putText(vis, str(num_circle), (int(tmpCNT[0][0][0]), int(tmpCNT[0][0][1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1)

            if num_circle >= NUMCIRCLE:
                good_point = self._find_good_points(tmp_cnt, corners, index_aruco)
                if good_point is not None:
                    effective_sign.append(val)
                    anchorPoint.append(np.array(good_point))
                    indexList.append(index_aruco)
                    self.idList.append(int(ids[index_aruco]))
                    self.prev_gray = frame_gray
                    print('successfully got!')
                else:
                    print('failed got!')
                    self.fcnt += 1

        return effective_sign, anchorPoint, indexList, rejectedImgPoints

    def _find_good_points(self, tmp_cnt, corners, index_aruco):
        for tmp_cnt_unit in combinations(tmp_cnt, 4):
            unit_list = list(tmp_cnt_unit)
            tmp_point = []
            for k2, cnt2 in enumerate(unit_list[0:NUMCIRCLE]):
                rect = cv2.minAreaRect(cnt2)
                box_ = cv2.boxPoints(rect)
                h = abs(box_[3, 1] - box_[1, 1])
                w = abs(box_[3, 0] - box_[1, 0])
                if h > 500 or w > 500:
                    continue
                tmp_point.append(list(rect[0]))
            
            if len(tmp_point) < NUMCIRCLE:
                continue
                
            cornerP = np.array(corners[int(index_aruco)][0, 0], dtype=int)
            final_point = sort_4_Vector(tmp_point, cornerP)
            
            flag_aruco_in = True
            for k in range(0, 4):
                if not is_pt_in_poly(corners[int(index_aruco)][0, k], final_point):
                    print('!!!!it is not have any aruco there!!!!!')
                    flag_aruco_in = False
                    break
            
            if flag_aruco_in:
                return final_point
        return None

    def _process_detection(self, vis, corners, ids, anchorPoint, indexList, frame_gray):
        content_total = ''
        self.FrameCnt += 1
        
        for index_val, val_acp in enumerate(anchorPoint):
            idx = int(ids[indexList[index_val]])

            [val_acp, self.old_val_acp[idx]] = filter_acp(val_acp, self.old_val_acp[idx], 
                                                          num_pixels=FILTER_THRESHOLD)
            val_acp.dtype = np.float
            
            rvecs = self.data_position_rvec[idx][1]
            tvecs = self.data_position_rvec[idx][2]
            rotation_vector_init = self.data_position_rvec[idx][1]
            translation_vector_init = self.data_position_rvec[idx][2]
            useExtrinsicGuess = True
            
            for k in range(0, len(val_acp)):
                cv2.putText(vis, str(k), (int(val_acp[k][0]), int(val_acp[k][1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1)
                cv2.circle(vis, (int(val_acp[k][0]), int(val_acp[k][1])), 1, (0, 0, 255), 1)
            
            for x, y in np.float32(val_acp).reshape(-1, 2):
                self.tracks.append([(x, y)])

            if len(rvecs) == 0 or len(tvecs) == 0:
                (success, rotation_vector, translation_vector) = cv2.solvePnP(
                    gt_3d_points, val_acp, self.cameraMatrix,
                    self.distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                err1 = 0
                print('init')
            else:
                transfom_set = cv2.solvePnPGeneric(gt_3d_points, val_acp, self.cameraMatrix,
                                                    self.distCoeffs,
                                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if transfom_set[0] != 0:
                    rotation_vector = transfom_set[1][0]
                    translation_vector = transfom_set[2][0]
                    err1 = transfom_set[3][0]
                    err2 = transfom_set[3][1]
                else:
                    print('calculate failed')
                    continue

                if (err1 >= 0.1 and err2 / err1 > 2.0) or (err2 / err1 <= 2.0):
                    print('recalculate!')
                    transfom_set = cv2.solvePnPGeneric(
                        gt_3d_points, val_acp, self.cameraMatrix,
                        self.distCoeffs, useExtrinsicGuess=useExtrinsicGuess,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                        rvec=rotation_vector_init, tvec=translation_vector_init)
                    rotation_vector = transfom_set[1][0]
                    translation_vector = transfom_set[2][0]
                    err1 = transfom_set[3][0]

            rvecs = rotation_vector
            tvecs = translation_vector
            meanrvec = rvecs

            print("RP", idx, ':', err1)

            time_this = time.time()
            if 0 != self.data_position_rvec[idx][3]:
                time_last = self.data_position_rvec[idx][3]
            else:
                time_last = np.copy(time_this)

            ball_center = self._calculate_ball_center(val_acp, tvecs)
            
            rvec2rmat = cv2.Rodrigues(meanrvec)
            pr = np.dot(radius_vec, rvec2rmat[0])

            self.data_position_rvec[idx][0] = np.copy(ball_center)
            self.data_position_rvec[idx][1] = np.copy(meanrvec)
            self.data_position_rvec[idx][2] = np.copy(tvecs)
            self.data_position_rvec[idx][3] = np.copy(time_this)
            
            content2 = '\nCamNo:{0}\nFrame:{1}\nidx:{2}\nbc:{3}\nrvec:{4}\ntvec:{5}\ntime:{6}'.format(
                str(CAMNO), str(self.FrameCnt), str(idx), repr(ball_center),
                repr(meanrvec.reshape(1, 3)), repr(tvecs.reshape(1, 3)), str(time_this))
            print(content2)
            content_total = content_total + content2

        self.socket.sendto(content_total.encode("utf-8"), (SERVERIP, bind_PORT))

    def _display_frame(self, vis):
        vis = vis.astype(np.uint8)
        cv2.putText(vis, str(CAMNO), (int(FIGWIDTH / 2), int(FIGHEIGHT / 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        vis = cv2.resize(vis, (240, 240))
        vis = cv2.flip(vis, -1)
        img = image.load(vis[..., ::-1].tobytes(), (240, 240))
        display.show(img, remote_show=False)

    def run(self, aruco_dict, parameters):
        cap = cv2.VideoCapture(0)
        cap.set(3, FIGWIDTH)
        cap.set(4, FIGHEIGHT)

        while True:
            ttoal = time.time()
            suc, raw_frame = cap.read()
            frame = raw_frame
            vis = frame.copy()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
            frame = frame.astype(np.uint8)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            flag_track, good = self._process_optical_flow(frame_gray, vis)
            
            if flag_track and good is not None:
                flag_track = self._process_tracking(vis, good, ttoal)
                
            if flag_track:
                vis = cv2.resize(vis, (240, 240))
                vis = cv2.flip(vis, -1)
                img = image.load(vis[..., ::-1].tobytes(), (240, 240))
                display.show(img, remote_show=False)
                continue

            self.idList.clear()
            ttoal = time.time()

            if len(corners) <= 0:
                print('can not find any aruco')
                self._display_frame(vis)
                continue

            effective_sign, anchorPoint, indexList, rejectedImgPoints = self._detect_markers(
                frame, frame_gray, corners, ids, vis)

            aruco.drawDetectedMarkers(vis, corners, ids)
            aruco.drawDetectedMarkers(vis, rejectedImgPoints)

            if len(effective_sign) > 0:
                self._process_detection(vis, corners, ids, anchorPoint, indexList, frame_gray)

            self._display_frame(vis)
            print('total', time.time() - ttoal)

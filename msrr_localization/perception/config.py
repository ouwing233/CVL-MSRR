import numpy as np
import cv2

radius_vec = np.array([0, 0, 32.5], dtype=float)
radius_vec2 = np.array([0, 0, 30.0], dtype=float)
width_gd = 18.
width_pix_g1 = 338
width_pix_g2 = 618
distance_gd = 150.
focal_length = distance_gd * width_pix_g1 / width_gd

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
send_PORT_default = 8888
bind_PORT = 9997

gt_3d_points = np.array(([-25, 25, 0], [25, 25, 0], [25, -25, 0], [-25, -25, 0]), dtype=np.double)

marker_num = 6
ept = -360. / marker_num
degree_list = {
    0: 0, 
    1: 1 * ept, 
    2: 2 * ept, 
    3: 3 * ept, 
    -1: -1 * ept, 
    -2: -2 * ept, 
    -3: -3 * ept
}

decay = 0.8

useExtrinsicGuess = True

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

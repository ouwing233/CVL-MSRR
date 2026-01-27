import threading
import cv2.aruco as aruco
import numpy as np

from .detector import ArucoDetector


def main():
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    parameters = aruco.DetectorParameters_create()
    np.set_printoptions(suppress=True)
    
    detector = ArucoDetector()
    
    t1 = threading.Thread(target=detector.run, args=(aruco_dict, parameters))
    tlist = [t1]
    
    try:
        for k in tlist:
            k.setDaemon(True)
        for k in tlist:
            k.start()
    except Exception as e:
        print(e)

    t1.join()


if __name__ == '__main__':
    main()

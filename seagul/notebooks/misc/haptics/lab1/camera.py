# taken from https://github.com/sgillen/vision_ucsb/blob/master/src/vision/src/camera_test.py

import cv2
import cv2.aruco        as Aruco
import numpy as np

# Which dictionary of markers are we using?
DICTIONARY  = Aruco.DICT_6X6_1000
# number of pixels on each side of the marker
MARKER_SIZE = 500

TEST_ARUCO_MARKER_GENERATION = 0
TEST_ARUCO_MARKER_DETECTION = 1

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("something went wrong! video not open")
    raise SystemExit


print("looks like the computer see's the camera, displaying image now")
print( "press q to move on to following tests")

cv2.namedWindow('test')
while(True):
    re, img = cap.read()
    cv2.imshow('test', img)


    # Exit if q is pressed
    if cv2.waitKey(1) == ord('q'):
        break


print()
print("moving on to ARUCO testing")
# load dictionary specified in config
aruco_Dict =  Aruco.Dictionary_get(DICTIONARY)

if TEST_ARUCO_MARKER_GENERATION:
    print()
    print("lets test aruco generation")
    img1 = Aruco.drawMarker( aruco_Dict, 2, MARKER_SIZE)
    img2 = Aruco.drawMarker( aruco_Dict , 3, MARKER_SIZE)
    cv2.imwrite("markertest1.jpg",img1)
    cv2.imwrite("markertest2.jpg",img2)
    #cv2.imshow('fhi1',img1)
    cv2.imshow('fhi2',img2)
    cv2.moveWindow('fhi2',1300,400)
    cv2.waitKey(0)

if TEST_ARUCO_MARKER_DETECTION:
    print()
    print("now testing for marker detection")
    print("checking for markers of libary ID:" ,DICTIONARY)

    parameters = Aruco.DetectorParameters_create()
    image = cv2.imread("board_aruco_57.png")
    corners, ids, rejectedImgPoints = Aruco.detectMarkers(image,aruco_Dict, parameters = parameters)
    print(corners,ids,rejectedImgPoints)
    found = Aruco.drawDetectedMarkers(image,corners,ids)
    cv2.imshow('f',found)
    reject = Aruco.drawDetectedMarkers(image, rejectedImgPoints, borderColor = (100,100,240))
    cv2.imshow('r',reject)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    video = 1
    if video:

        re, img = cap.read()
        cv2.namedWindow('Raw Video')
        cv2.imshow('Raw Video', img)

        cv2.namedWindow('Markers')
        cv2.imshow('Markers', img)
        cv2.moveWindow('Markers', 700, 0)

        cv2.namedWindow('Rejects')
        cv2.imshow('Rejects', img)
        cv2.moveWindow('Rejects', 1300, 0)

        while(True):
            re, img = cap.read()
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = Aruco.detectMarkers(gray, aruco_Dict, parameters = parameters)
            print(corners,ids,rejectedImgPoints)
            img2 = np.copy(img)
            cv2.imshow('Raw Video',img)
            found = Aruco.drawDetectedMarkers(img,corners,ids)
            cv2.imshow('Markers',found)
            reject = Aruco.drawDetectedMarkers(img2,rejectedImgPoints,borderColor = (100,100,240))
            cv2.imshow('Rejects',reject)

            if cv2.waitKey(1) == ord('q'):
                break
import cv2
import pyrealsense2 as rs 
import numpy as np
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
# cap2.set(cv2.CAP_PROP_AUTOFOCUS, 0)

num = 0

while cap.isOpened():

    succes1, img = cap.read()
    succes2, img2 = cap2.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        # cv2.imwrite('R_test'+str(num)+'.jpg', img)
        
        # # # real_sense
        # cv2.imwrite('L_test'+str(num)+'.jpg', img2)

        cv2.imwrite('stereo_check/R/'+str(num)+'.jpg', img)
        
        # real_sense
        cv2.imwrite('stereo_check/L/'+str(num)+'.jpg', img2)
        
        print("images saved!")
        num += 1

    cv2.imshow('right',img)
    cv2.imshow('left',img2)
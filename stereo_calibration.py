import numpy as np
import cv2 as cv
import glob

def stereo_calibration(): 
    chessboardSize = (8, 6)
    frameSize = (640, 480)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2) * 26

    objpoints = []
    imgpointsL = [] 
    imgpointsR = []     
    
    # images path
    images_left = glob.glob('stereo_check/L/*.jpg')
    images_right = glob.glob('stereo_check/R/*.jpg')

    # make chesspattern 
    for imgLeft, imgRight in zip(images_left, images_right):
        imgL = cv.imread(imgLeft)
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        imgR = cv.imread(imgRight)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
        retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

        # read image 
        if retL == True:
            objpoints.append(objp)
            cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
            imgpointsL.append(cornersL)
            cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            imgpointsR.append(cornersR)
            
            cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
            # cv.imshow('img left', imgL)
            cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
            # cv.imshow('img right', imgR)
            # cv.waitKey(200)
        cv.destroyAllWindows()
    retL, matL, distL, rotL, transL, = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
    retR, matR, distR, rotR, transR, = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)

    imgL = cv.imread('L_0.jpg')
    h,  w = imgL.shape[:2]
    newcameramtxL, roi = cv.getOptimalNewCameraMatrix(matL, distL, (w,h), 1, (w,h))
    img_undistL = cv.undistort(imgL, matL, distL, None, newcameramtxL)
    
    imgR = cv.imread('R_0.jpg')
    h,  w = imgR.shape[:2]
    newcameramtxR, roi = cv.getOptimalNewCameraMatrix(matR, distR, (w,h), 1, (w,h))
    img_undistR = cv.undistort(imgR, matR, distR, None, newcameramtxR)
    # cv.imshow('undistort L', img_undistL)
    # cv.imshow('undistort R', img_undistR)
    # cv.waitKey(10000)
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newcameramtxL, distL, newcameramtxR, distR, grayL.shape[::-1], criteria_stereo, flags)

    # undistort image L, R  // new camera matrix L, R // rot // trans  
    return img_undistL, img_undistR, newcameramtxL, newcameramtxR, rot, trans

if __name__ == '__main__' :
    img_undistL, img_undistR, newcameramtxL, newcameramtxR, rot, trans = stereo_calibration() 

    print ('------new camera matrix left------ \n',newcameramtxL)
    print ('------new camera matrix right------ \n',newcameramtxR)
    print ('------rotation matrix------ \n', rot)
    print ('------translation vector------ \n', trans)



import cv2
import numpy as np
import os

def croppImages(path, x, y, w, h):
    # get all calibration images
    images = []
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path,f)):
            if f.endswith(".jpg"):
                images.append(os.path.join(path,f))

    for image in images:    
        img = cv2.imread(image)
        # cv2.imshow('img', img)
        img_ = img[y:y+h, x:x+w]
        # cv2.imshow('img_', img_)
        cv2.imwrite(f"{image[:-4]}_cropped.png", img_)

def calibrate(path):
    """ path where calibration pictures are stored"""
    chessboardSize = (19,19)
    frameSize = (2332,2403)

    # termination criteria
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)


    size_of_chessboard_squares_mm = 6
    objp = objp * size_of_chessboard_squares_mm

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # get all calibration images
    images = []
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path,f)):
            images.append(os.path.join(path,f))

    for image in images:    
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _,binary = cv2.threshold(cv2.GaussianBlur(gray,(5,5),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(binary, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            print(image)
            objpoints.append(objp)
            # corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # cv2.drawChessboardCorners(img, chessboardSize, corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(1000)


    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, frameSize, 1, frameSize)
    return cameraMatrix, dist, newCameraMatrix, roi


def calibrateImages(src_path,dst_path,cameraMatrix, dist, newCameraMatrix, roi):
    """ calibrate all images in path and store them"""
    images = []
    names = []
    for f in os.listdir(src_path):
        if os.path.isfile(os.path.join(src_path,f)):
            images.append(os.path.join(src_path,f))
            names.append(f)
    
    for image,idx,name in zip(images,range(len(images)),names): 
        im = cv2.imread(image)
        dst = cv2.undistort(im, cameraMatrix, dist, None, newCameraMatrix)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w+400]
        cv2.imwrite(os.path.join(dst_path,f"{name}_calib.png"), dst)
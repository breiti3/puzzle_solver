import os
from calibration.calibration import calibrate, calibrateImages, croppImages
import cv2


CALIBRATION_PATH = os.path.join("C:",os.path.sep,"Users","TV","Pictures","Puzzel_3000_Pieces","Calibration_1")
PICTURES_PATH = os.path.join("C:",os.path.sep,"Users","TV","Pictures","Puzzel_3000_Pieces","Pictures_1")
CALIB_PICTURES_PATH = os.path.join("C:",os.path.sep,"Users","TV","Pictures","Puzzel_3000_Pieces","Calibrated_Pictures_1")



def main():
    cameraMatrix, dist, newCameraMatrix, roi = calibrate(CALIBRATION_PATH)
    calibrateImages(PICTURES_PATH,CALIB_PICTURES_PATH,cameraMatrix, dist, newCameraMatrix, roi)








if __name__ == "__main__":
    main()


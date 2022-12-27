import os
from calibration.calibration import calibrate, calibrateImages, croppImages
import cv2


CALIBRATION_PATH = os.path.join("C:",os.path.sep,"Users","TV","Pictures","Puzzel_3000_Pieces","Calibration_1")
PICTURES_PATH = os.path.join("C:",os.path.sep,"Users","TV","Pictures","Puzzel_3000_Pieces","Pictures_1")
CALIB_PICTURES_PATH = os.path.join("C:",os.path.sep,"Users","TV","Pictures","Puzzel_3000_Pieces","Calibrated_Pictures_1")

WORKING_FOLDER = os.path.join("C:",os.path.sep,"Users","TV","Pictures","Puzzel_3000_Pieces","Processed_1")

def main():
    cropp_images()
    cameraMatrix, dist, newCameraMatrix, roi = calibrate(os.path.join(WORKING_FOLDER,"calib"))
    calibrateImages(os.path.join(WORKING_FOLDER,"images"),CALIB_PICTURES_PATH,cameraMatrix, dist, newCameraMatrix, roi)


def cropp_images():
    # DEFINE ME
    x = 4000-3336
    y = 0
    w = 3139
    h = 3000

    try:
        os.mkdir(os.path.join(WORKING_FOLDER,"calib"))
    except:
        pass
    for f in os.listdir(CALIBRATION_PATH):

        if os.path.isfile(os.path.join(CALIBRATION_PATH,f)):
                im = cv2.imread(os.path.join(CALIBRATION_PATH,f))
                im = im[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(WORKING_FOLDER,"calib",f),im)

    try:
        os.mkdir(os.path.join(WORKING_FOLDER,"images"))
    except:
        pass
    for f in os.listdir(PICTURES_PATH):

        if os.path.isfile(os.path.join(PICTURES_PATH,f)):
                im = cv2.imread(os.path.join(PICTURES_PATH,f))
                im = im[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(WORKING_FOLDER,"images",f),im)




if __name__ == "__main__":
    main()


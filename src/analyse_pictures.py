import os
from plot_helper.plot_helper import PlotPieces
from calibration.calibration import calibrate, calibrateImages, croppImages
from data_management.DataManager import DataManager
import pickle
from calc.ErrorCalculator import calc_error_and_rot_mat
import matplotlib.pyplot as plt


CALIB_PICTURES_PATH = os.path.join("C:",os.path.sep,"Users","TV","Pictures","Puzzel_3000_Pieces","Calibrated_Pictures_3")

def main():
    path = CALIB_PICTURES_PATH
    piecesListPath = os.path.join(path,"pieces.pickle")
    errorDictPath = os.path.join(path,"error_dict.pickle")
    m = DataManager()
    plotter = PlotPieces()
    if os.path.exists(piecesListPath):
        pieces = m.loadPieces(piecesListPath)
    else:
        pieces = m.generatePieces(path)
        m.storePieces(piecesListPath, pieces)

    plotter.plot_all(pieces)
    plt.show()
    if os.path.exists(errorDictPath):
        with open(errorDictPath,"rb") as f:
            obj = pickle.load(f)
        error = obj["error"]
        transformMat = obj["transformMat"]
    else:
        error,transformMat = calc_error_and_rot_mat(pieces)
        obj = {"error":error,"transformMat":transformMat}
        with open(errorDictPath,"wb") as f:
            pickle.dump(obj,f)






if __name__ == "__main__":
    main()
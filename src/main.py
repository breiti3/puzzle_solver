import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from calc.SinglePiece import SinglePiece
from calc.PieceCompare import box_compare, type_compare, icp_compare
# from scalene import scalene_profiler
from plot_helper.plot_helper import PlotPieces
from calibration.calibration import calibrate, calibrateImages, croppImages
from data_management.DataManager import DataManager
from data_management.ClusterManager import ClusterManager
from calc.ErrorCalculator import calc_error_and_rot_mat
import pickle

# def calibrate(path):
#     chessboardSize = (19,19)
#     frameSize = (2332,2403)

#     # termination criteria
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#     # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#     objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
#     objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)


#     size_of_chessboard_squares_mm = 6
#     objp = objp * size_of_chessboard_squares_mm

#     # Arrays to store object points and image points from all the images.
#     objpoints = [] # 3d point in real world space
#     imgpoints = [] # 2d points in image plane.

#     img = cv2.imread(path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#         corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners)


#         ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
#         newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, frameSize, 1, frameSize)
#         return cameraMatrix, dist, newCameraMatrix, roi
def main():
    path = os.path.join("..","..","puzzle_bilder","With Calibration Pattern","pieces_cut.jpg")
    calibPath = os.path.join("..","..","puzzle_bilder","With Calibration Pattern","chessboard_cut.jpg")

    cameraMatrix, dist, newCameraMatrix, roi = calibrate(calibPath)

    minArea = 50000
    maxArea = 250000
    im = cv2.imread(path)
    # Undistort
    dst = cv2.undistort(im, cameraMatrix, dist, None, newCameraMatrix)

    # crop the image
    x, y, w, h = roi
    im = dst[y:y+h, x:x+w]

    np.set_printoptions(linewidth=300,precision=0)

    gray = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)
    _,binary = cv2.threshold(cv2.GaussianBlur(gray,(15,15),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    pieces = []
    for c in contours:
        pieces.append(SinglePiece(c,minArea=minArea,maxArea=maxArea))

    pieces = [p for p in pieces if p.valid]
    area = np.array([p.area for p in pieces])

    # idx = area.argsort()[::-1]
    # pieces = [pieces[i] for i in idx]

    error = np.zeros((len(pieces)*4,len(pieces)*4))
    error[:] = np.inf
    rotMat = np.zeros((len(pieces)*4,len(pieces)*4,3,3))
    for i in range(len(pieces)):
        for j in range(i+1,len(pieces)):
            error[4*i:4*i+4,4*j:4*j+4] = 0
            error[4*i:4*i+4,4*j:4*j+4] = type_compare(pieces[i],pieces[j])
            error[4*i:4*i+4,4*j:4*j+4] = box_compare(pieces[i],pieces[j],maxPixelDelta=18*2,matrix=error[4*i:4*i+4,4*j:4*j+4])#18 pixel is about 1mm
            error[4*i:4*i+4,4*j:4*j+4],rotMat[4*i:4*i+4,4*j:4*j+4,:,:] = icp_compare(pieces[i],pieces[j], maxError = 20,matrix=error[4*i:4*i+4,4*j:4*j+4])#18 pixel is about 1mm

    i_lower = np.tril_indices(error.shape[0], -1)
    error[i_lower] = error.T[i_lower]
    tmp = np.transpose(rotMat,(1,0,2,3))
    for i1,i2 in zip(i_lower[0],i_lower[1]):
        if tmp[i1,i2,2,2] == 1:
            rotMat[i1,i2,:,:] = np.linalg.inv(tmp[i1,i2,:,:])
            rotMat[i1,i2,2,:] = np.array([0,0,1])






    # plot some pieces
    e_ = error
    rotHist = []
    plotter = PlotPieces()
    tmp = np.unravel_index(np.argmin(e_, axis=None), e_.shape)
    idx = np.floor(np.array(tmp)/4).astype(int)
    # clear these two peaces
    e_[4*idx[0]:4*idx[0]+4,4*idx[1]:4*idx[1]+4] = np.inf
    e_[4*idx[1]:4*idx[1]+4,4*idx[0]:4*idx[0]+4] = np.inf

    rotHist.append(tmp)
    plt.figure()
    plotter.plot_piece(pieces[idx[0]],idx[0])
    plotter.rotate_and_plot(rotMat,rotHist,pieces[idx[1]],idx[1])
    print(f"Match {idx[0]} with {idx[1]}")

    for ii in range(3):
        tmp = np.unravel_index(np.argmin(e_[4*idx[1]:4*idx[1]+4,:], axis=None)+4*idx[1]*e_.shape[1], e_.shape)
        rotHist.append(tmp)
        idx = np.floor(np.array(tmp)/4).astype(int)
        # clear these two peaces
        e_[4*idx[0]:4*idx[0]+4,4*idx[1]:4*idx[1]+4] = np.inf
        e_[4*idx[1]:4*idx[1]+4,4*idx[0]:4*idx[0]+4] = np.inf

        print(f"Match {idx[0]} with {idx[1]}")
        plotter.rotate_and_plot(rotMat,rotHist,pieces[idx[1]], idx[1])



    plt.figure()
    plt.subplot(121)
    plt.imshow(im)
    axs=plt.subplot(122)
    axs.set_aspect('equal', 'box')
    colors = ["r","b"]
    for p,i in zip(pieces,range(len(pieces))):
        for t,s in zip(p.edgeType,p.sides):
            plt.plot(s[:,0,0],s[:,0,1],colors[int((t+1)/2)])
        for e in p.edgeIdx:
            plt.plot(p.points[e,0,0],p.points[e,0,1],'rx', markersize=12)

        plt.text(np.mean(p.points[:,0,0]),np.mean(p.points[:,0,1]),f'{i}, A= {area[i]}')
    # plt.gca().invert_yaxis()
    plt.show()
    pass


def main2():
    path = os.path.join("..","..","puzzle_bilder","Run_1")

    cameraMatrix, dist, newCameraMatrix, roi = calibrate(path)
    calibrateImages(path,cameraMatrix, dist, newCameraMatrix, roi)

def main3():
    path = os.path.join("..","..","puzzle_bilder","Run_1")
    croppImages(path,600,300,3400,2700)


def main4():
    path = os.path.join("..","..","puzzle_bilder","Run_1","CalibratedPictures")
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
    # plt.show()

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
    

    solvePuzzle(pieces,error,transformMat)

    # clusterList = []
    # clusterMatrix = []

   
    # e_ = error
    # rotHist = []
    # plotter = PlotPieces()

    # tmp = np.unravel_index(np.argmin(e_, axis=None), e_.shape)
    # idx = np.floor(np.array(tmp)/4).astype(int)

    # # get clusters
    # cluster = [None,None]
    # for c in clusterList:
    #     for i in range(2):
    #         if cluster[i] == None:
    #             if idx[i] in c:
    #                 cluster[i] = c

    # # check all cases
    # if cluster == [None,None]:
    #     # no cluster exists, create new
    #     clusterList.append(idx)


    # # remove these two side, since they are used now
    # e_[tmp[0],:]= np.inf
    # e_[:,tmp[1]]= np.inf
    # # clear these two peaces
    # e_[4*idx[0]:4*idx[0]+4,4*idx[1]:4*idx[1]+4] = np.inf
    # e_[4*idx[1]:4*idx[1]+4,4*idx[0]:4*idx[0]+4] = np.inf

    # rotHist.append(tmp)
    # plt.figure()
    # plotter.plot_piece(pieces[idx[0]])
    # plotter.rotate_and_plot(transformMat,rotHist,pieces[idx[1]])
    # print(f"Match {idx[0]} with {idx[1]}")

    # for ii in range(1):
    #     tmp = np.unravel_index(np.argmin(e_[4*idx[1]:4*idx[1]+4,:], axis=None)+4*idx[1]*e_.shape[1], e_.shape)
    #     rotHist.append(tmp)
    #     idx = np.floor(np.array(tmp)/4).astype(int)

    #     # remove these two side, since they are used now
    #     e_[tmp[0],:]= np.inf
    #     e_[:,tmp[1]]= np.inf
    #     # clear these two peaces
    #     e_[4*idx[0]:4*idx[0]+4,4*idx[1]:4*idx[1]+4] = np.inf
    #     e_[4*idx[1]:4*idx[1]+4,4*idx[0]:4*idx[0]+4] = np.inf

    #     print(f"Match {idx[0]} with {idx[1]}")
    #     plotter.rotate_and_plot(transformMat,rotHist,pieces[idx[1]])
    # plt.axis('equal')
    # plt.show()
    pass


def solvePuzzle(pieces,error,transformMat,treshold=3):
    np.set_printoptions(linewidth=300,precision=0)

    # init stuff
    clusterList:list[ClusterManager] = []
    e_ = error

    e_values = e_[np.invert(np.isinf(e_))]
    # counts, bins = np.histogram(e_values,100)
    # plt.stairs(counts, bins)
    # idx
    tmp = np.unravel_index(np.argmin(e_, axis=None), e_.shape)
    v = e_[tmp]

    if v > treshold:
        print(f"Error {v} bigger than treshold {treshold}, exit before start solving")
        return
    idx = np.floor(np.array(tmp)/4).astype(int)
    side_idx = np.remainder(tmp,4)

    # remove these two side, since they are used now
    e_[tmp[0],:]= np.inf
    e_[tmp[1],:]= np.inf
    e_[:,tmp[0]]= np.inf
    e_[:,tmp[1]]= np.inf
    # clear these two peaces, these are already togheter
    e_[4*idx[0]:4*idx[0]+4,4*idx[1]:4*idx[1]+4] = np.inf
    e_[4*idx[1]:4*idx[1]+4,4*idx[0]:4*idx[0]+4] = np.inf

    clusterList.append(ClusterManager(idx[0]))
    clusterList[-1].addPiece(dst_piece_idx=idx[0],dst_side_idx=side_idx[0],src_piece_idx=idx[1],src_side_idx=side_idx[1])

    # repeat
    while True:
        tmp = np.unravel_index(np.argmin(e_, axis=None), e_.shape)
        v = e_[tmp]
        if v > treshold:
            break 

        idx = np.floor(np.array(tmp)/4).astype(int)
        side_idx = np.remainder(tmp,4)

        # remove these two side, since they are used now
        e_[tmp[0],:]= np.inf
        e_[tmp[1],:]= np.inf
        e_[:,tmp[0]]= np.inf
        e_[:,tmp[1]]= np.inf
        # clear these two peaces, these are already togheter
        e_[4*idx[0]:4*idx[0]+4,4*idx[1]:4*idx[1]+4] = np.inf
        e_[4*idx[1]:4*idx[1]+4,4*idx[0]:4*idx[0]+4] = np.inf

        # check for existing clusters
        cluster = [None,None]
        for c in clusterList:
            for i in range(2):
                if cluster[i] == None:
                    exists,_=c.checkForPiece(idx[i])
                    if exists:
                        cluster[i] = c

        if all([c == None for c in cluster]):
            # no cluster exists, create new
            clusterList.append(ClusterManager(idx[0]))
            clusterList[-1].addPiece(dst_piece_idx=idx[0],dst_side_idx=side_idx[0],src_piece_idx=idx[1],src_side_idx=side_idx[1])
        elif any([c == None for c in cluster]):
            # only one is none
            for i in range(2):
                if not (cluster[i] == None):
                    cluster[i].addPiece(dst_piece_idx=idx[i],dst_side_idx=side_idx[i],src_piece_idx=idx[(i+1)%2],src_side_idx=side_idx[(i+1)%2])
        else:
            # all are not none
            if cluster[0] is cluster[1]:
                print("Error clusters are the same")
            # add cluster to cluster
            cluster[0].addCluster(cluster[1],dst_piece_idx=idx[0],dst_side_idx=side_idx[0],src_piece_idx=idx[1],src_side_idx=side_idx[1])
            # remove cluster 1, it is now in cluster 0
            clusterList.remove(cluster[1])


    # plot the stuff
    plotter = PlotPieces()
    for c in clusterList:
        plt.figure()
        transformInfo = c.generate_transformation_info(transformMat)
        for k,v in transformInfo.items():
            plotter.transform_and_plot(v,pieces[k])


    plt.axis('equal')
    plt.show()


    pass
        

if __name__ == "__main__":
    # scalene_profiler.start()
    # main()
    # main3()
    # main2()
    
    main4()
    # scalene_profiler.stop()
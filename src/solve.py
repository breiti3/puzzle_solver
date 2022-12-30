import numpy as np
import os
from data_management.DataManager import DataManager
from data_management.ClusterManager import ClusterManager
from plot_helper.plot_helper import PlotPieces
import matplotlib.pyplot as plt
import pickle
from calc.ErrorCalculator import calc_error_and_rot_mat


PICKLE_PATH = os.path.join("C:",os.path.sep,"Users","ericb","Documents","Git_workspace","puzzle_bilder","firstRun")

def main():
    piecesListPath = os.path.join(PICKLE_PATH,"pieces.pickle")
    errorDictPath = os.path.join(PICKLE_PATH,"error_dict.pickle")
    m = DataManager()
    plotter = PlotPieces()
    if os.path.exists(piecesListPath):
        pieces = m.loadPieces(piecesListPath)
    else:
        print(f"No pieces pickle found. Exit...")
        return


    if os.path.exists(errorDictPath):
        with open(errorDictPath,"rb") as f:
            obj = pickle.load(f)
        error = obj["error"]
        transformMat = obj["transformMat"]
    else:
        print(f"No error matrix pickle found. Exit...")
        return
    

    solvePuzzle(pieces,error,transformMat)


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

        

if __name__ == "__main__":
    main()
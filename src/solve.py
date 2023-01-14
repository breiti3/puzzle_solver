import numpy as np
import os
from data_management.DataManager import DataManager
from data_management.ClusterManager import ClusterManager
from plot_helper.plot_helper import PlotPieces
import matplotlib.pyplot as plt
import pickle
from calc.ErrorCalculator import calc_error_and_rot_mat
import json

PICKLE_PATH = os.path.join("C:",os.path.sep,"Users","ericb","Documents","Git_workspace","puzzle_bilder","secondRun")
MAX_NUMBER_OF_PLOTS = 4
def main():
    piecesListPath = os.path.join(PICKLE_PATH,"pieces.pickle")
    errorDictPath = os.path.join(PICKLE_PATH,"error_dict.pickle")
    exeptionListPath = os.path.join(PICKLE_PATH,"exeption_list.json")
    solved_listPath = os.path.join(PICKLE_PATH,"solved_list.json")
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

    if os.path.exists(exeptionListPath):
        with open(exeptionListPath,"r") as f:
            data = json.load(f)
        exeptionList = data["exeptionList"]
    else:
        exeptionList = []

    if os.path.exists(solved_listPath):
        with open(solved_listPath,"r") as f:
            data = json.load(f)
        solved_list = data["solved_list"]
    else:
        solved_list = []

    
    
    for p in pieces:
        plt.figure(p.pictureName)
        plt.plot(p.points[:,0,0],p.points[:,0,1])
        plt.text(np.mean(p.points[:,0,0]),np.mean(p.points[:,0,1]),f"{p.idx}",size=8)
        plt.plot()
    solvePuzzle(pieces,error,transformMat,treshold=4,execptionList=exeptionList,solved_list=solved_list)


def solvePuzzle(pieces,error,transformMat,treshold=3, execptionList=[],solved_list=[]):
    np.set_printoptions(linewidth=300,precision=0)

    solved_picture = [x[0] for x in solved_list]
    solved_piece = [x[1] for x in solved_list]

    # init stuff
    clusterList:list[ClusterManager] = []
    e_ = error

    for i,j in execptionList:
        e_[i,j] = np.inf
        e_[j,i] = np.inf

    # e_values = e_[np.invert(np.isinf(e_))]
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
    # clusterList = sorted(clusterList,key=lambda x:x.number_of_pieces,reverse=True)
    clusterList = sorted(clusterList,key=lambda x:x.number_of_pieces)
    nplots = 0
    for c in clusterList:
        plt.figure()
        allPiecesSolved = True
        for p in c.piece_idx_list:
            found = False
            for piece_picture_name,piece_number in zip(solved_picture,solved_piece):
                if ((int(pieces[p].pictureName.replace(".png","")) == piece_picture_name) and (pieces[p].idx == piece_number)):
                    found = True
                    break
            if not found:
                allPiecesSolved = False


        if not allPiecesSolved:
            nplots += 1
            transformInfo = c.generate_transformation_info(transformMat)
            for k,v in transformInfo.items():
                plotter.transform_and_plot(v,pieces[k],k)
            if nplots >= MAX_NUMBER_OF_PLOTS:
                break


    print(f"From {len(pieces)} pieces, {sum([c.number_of_pieces for c in clusterList])} are in a connection")
    plt.axis('equal')
    plt.show()
    pass

        

if __name__ == "__main__":
    main()
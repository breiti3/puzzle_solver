from calc.SinglePiece import SinglePiece
import numpy as np
import sklearn.neighbors
import matplotlib.pyplot as plt
import cv2
from numba import jit

def box_compare(srcPiece:SinglePiece,trgPiece:SinglePiece, maxPixelDelta = 100, matrix = None) -> float:
    # return np.inf if box sides are more than maxPixelDelta different in size, else 0
    # pass an already calculated matrix in. The fields with 0 will be box compared
    error = _compare_boxes(srcPiece.normalized_points[srcPiece.edgeIdx,0,:],trgPiece.normalized_points[trgPiece.edgeIdx,0,:],maxPixelDelta)
    if matrix is not None:
        error[(matrix == np.inf)] = np.inf
    return error

def type_compare(srcPiece:SinglePiece,trgPiece:SinglePiece, matrix = None):
    error = _compare_type(srcPiece.edgeType,trgPiece.edgeType)
    if matrix is not None:
        error[(matrix == np.inf)] = np.inf
    return error


def icp_compare(srcPiece:SinglePiece,trgPiece:SinglePiece, maxError = 50, matrix = None) -> float:
    if matrix is not None:
        return _compare_icp(srcPiece.normalized_sides,trgPiece.normalized_sides,maxError, matrix)
    else:
        error = np.zeros((4,4))
        return _compare_icp(srcPiece.normalized_sides,trgPiece.normalized_sides,maxError, error)


def _compare_type(src,trg):
    error = np.zeros((4,4))
    tmp = np.matmul(src[:,None],trg.reshape((-1,1)).T)
    error[tmp>0] = np.inf
    return error

def _compare_boxes(src,trg,maxPixelDelta):
    error = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            e = np.abs(np.sqrt(np.sum((src[i,:]-src[(i+1)%4,:])**2))-np.sqrt(np.sum((trg[j,:]-trg[(j+1)%4,:])**2)))
            if e > maxPixelDelta:
                error[i,j] = np.inf
    return error


def _compare_icp(srcPiece,trgPiece,maxError,error):
    transformMat = np.zeros((4,4,3,3))
    for i in range(4):
        for j in range(4):
            if error[i,j] != np.inf:
                ss = srcPiece[i]
                ts = trgPiece[j]
                # plt.figure()
                # plt.plot(ss[:,0,0],ss[:,0,1],'b')
                # plt.plot(ts[:,0,0],ts[:,0,1],'r')

                # find rotation angle
                a1 = np.arctan2(ss[-1,0,1]-ss[0,0,1], ss[-1,0,0]-ss[0,0,0])
                a2 = np.arctan2(ts[-1,0,1]-ts[0,0,1], ts[-1,0,0]-ts[0,0,0])+np.pi # add pi
                da = a1-a2 # delta angle

                R = np.array(((np.cos(da), -np.sin(da)), (np.sin(da), np.cos(da))))
                tsRot = np.dot(R,ts[:,0,:].T).T
                T = ((ss[-1,0,:]+ss[0,0,:])/2-(tsRot[0,:]+tsRot[-1,:])/2)

                # transformation matrix
                Tr = np.array([[R[0,0],R[0,1],T[0]],[R[1,0],R[1,1],T[1]],[0,0,1]])

                tsRotTrans = cv2.transform(ts, Tr[:2,:])
                # plt.plot(tsRotTrans[:,0,0],tsRotTrans[:,0,1],'r')


                dst = ss[:,0,:].T      
                src = tsRotTrans[:,0,:].T
                if dst.shape[1]<src.shape[1]:
                    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto').fit(src.T)
                    distances, indices = nbrs.kneighbors(dst.T)
                    dst_ = dst
                    src_ = src[:,indices[:,0]]
                else:
                    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst.T)
                    distances, indices = nbrs.kneighbors(src.T)
                    dst_ = dst[:,indices[:,0]]
                    src_ = src

                for ii in range(5):
                    H = _calc_rotation_and_translation(src_,dst_)
                    src = np.squeeze(cv2.transform(src[:,None,:].T, H)).T
                    Tr = (np.matrix(np.vstack((H,[0,0,1])))*np.matrix(Tr)).A


                    if dst.shape[1]<src.shape[1]:
                        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto').fit(src.T)
                        distances, indices = nbrs.kneighbors(dst.T)
                        src_ = src[:,indices[:,0]]
                    else:
                        distances, indices = nbrs.kneighbors(src.T)
                        dst_ = dst[:,indices[:,0]]
                        src_ = src
                    # m_dst = np.mean(dst2,axis=1,keepdims=True)

                    e = np.sum(distances**2)/len(distances)

                # plt.plot(src[0,:],src[1,:],'g')
                # test = cv2.transform(ts, Tr[:2,:])
                # plt.plot(test[:,0,0],test[:,0,1],'m')

                # check if edges are close enough together

                if maxError < e:
                    error[i,j] = np.inf
                else:
                    error[i,j] = e
                    transformMat[i,j,:,:] = Tr
                    # if dst.shape[1]<src.shape[1]:
                    #     print(e)

    return error,transformMat

@jit(nopython=True,cache=True)
def _calc_rotation_and_translation( src, dst):
    """ Get rotation and translation to get src to the dst location """
    m_dst = np.array([[np.mean(dst[0,:])],[np.mean(dst[1,:])]])
    dst_ = dst-m_dst
    # get center of src
    m_scr = np.array([[np.mean(src[0,:])],[np.mean(src[1,:])]])
    scr_ = src-m_scr
    H = scr_@dst_.T
    u,s,vh = np.linalg.svd(H)
    R = vh.T@u.T
    Tr_ = m_dst-R@m_scr
    H = np.concatenate((R,Tr_),axis=1)

    # return transformation and transformed src
    return H

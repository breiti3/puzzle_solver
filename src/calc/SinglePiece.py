import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import sys
from numba import jit


class SinglePiece():
    def __init__(self, points:np.array, minArea = 0, maxArea = sys.float_info.max, pictureSrcPath:str="",pictureName:str=""):
        self.points = points
        self.area = cv2.contourArea(points)
        if self.area < minArea or self.area > maxArea:
            self.valid = False
            return

        points = rotation_correction(points)
        self.center = np.mean(points,axis=0)
        self.normalized_points = (points - np.mean(points,axis=0)).astype(np.float32)
        self.edgeIdx = self._findEdges(np.squeeze(self.normalized_points),32)
        self.pictureSrcPath = pictureSrcPath
        self.pictureName = pictureName
        self.idx = 0
        self.sideIdx = [0,0,0,0]

        if len(self.edgeIdx) == 4:
            self.valid = True
            try:
                self.normalized_sides = get_sides(self.normalized_points, self.edgeIdx)
                self.sides = get_sides(self.points, self.edgeIdx)
                self.edgeType = get_edgeType(self.normalized_sides)
            except IndexError:
                self.valid = False
        else:
            self.valid = False


    def setIdx(self,idx:int):
        self.idx = idx

    def _findEdges(self,points,delta):
        (error,p) = calcErrorForEdges(points,delta)
        indexes, _ = scipy.signal.find_peaks(-error,distance=int(2*delta))
        return refineEdges(p,delta,error,indexes,points.shape[0])

def set_side_idx(self,side,idx):
    if side > len(self.sideIdx):
        return
    self.sideIdx[side]=idx

@jit(nopython=True,cache=True)
def rotation_correction( points):
    angle = np.arctan2(points[:,0,1],points[:,0,0])
    dangle = np.sum(angle[1:]-angle[:-1])
    if dangle < 0:
        points = points[::-1,:,:]
    return points

@jit(nopython=True,cache=True)
def refineEdges(p,delta,error,indexes,n_points):
    error_ = error[indexes]
    sI_ = error_.argsort()    

    if len(sI_) > 3:
        indexes = indexes[sI_[0:4]]
        edgeIdx = indexes[indexes.argsort()]
        # refine edge detection
        # plt.figure()
        for idx,i in zip(edgeIdx,range(len(edgeIdx))):
            idxOld = idx
            r = p[idx:idx+2*delta,0]**2+p[idx:idx+2*delta,1]**2
            edgeIdx[i] = idx + (np.argmax(r)-delta)
        return edgeIdx % n_points
    else:
        [0]

@jit(nopython=True,cache=True)
def calcErrorForEdges(points,delta):
    # generate vectors. 
    p3 = np.concatenate((points[delta:],points[:delta]))
    p1 = np.concatenate((points[-delta:],points[:-delta]))
    p = np.concatenate((points[-delta:],points,points[:delta]))
    v1 = points-p1
    v2 = points-p3
    # calculate angles from vector before and vector after to midpoint
    a1,a2 = corner_angle(v1,v2,points)

    # error function
    return ((a1-np.pi/4)**2+(a2-np.pi/4)**2),p

def findEdges(points,delta):
    # generate vectors. 
    p3 = np.concatenate((points[delta:],points[:delta]))
    p1 = np.concatenate((points[-delta:],points[:-delta]))
    p = np.concatenate((points[-delta:],points,points[:delta]))
    v1 = points-p1
    v2 = points-p3
    # calculate angles from vector before and vector after to midpoint
    a1,a2 = corner_angle(v1,v2,points)

    # error function
    error = (a1-np.pi/4)**2+(a2-np.pi/4)**2

    # find 4 lowest peaks -> this are the edges
    indexes, _ = scipy.signal.find_peaks(-error,distance=int(2*delta))
    error_ = error[indexes]
    sI_ = error_.argsort()    

    if len(sI_) > 3:
        indexes = indexes[sI_[0:4]]
        edgeIdx = indexes[indexes.argsort()]
        # refine edge detection
        # plt.figure()
        for idx,i in zip(edgeIdx,range(len(edgeIdx))):
            idxOld = idx
            r = p[idx:idx+2*delta,0]**2+p[idx:idx+2*delta,1]**2
            edgeIdx[i] = idx + (np.argmax(r)-delta)

            # plt.subplot(121)
            # plt.plot(r,'r')
            # plt.subplot(122)
            # plt.plot(points[idxOld-delta:idxOld+delta,0],points[idxOld-delta:idxOld+delta,1],'rx-')
            # plt.plot(points[idxOld,0],points[idxOld,1],'bx')
            # plt.plot(points[edgeIdx[i],0],points[edgeIdx[i],1],'gx')
            pass


        return edgeIdx
    else:
        []

    # plt.figure()
    # plt.subplot(221)
    # plt.plot(points[:,0],points[:,1],"*-")

    # for i in idx:
    #     plt.plot(points[i,0],points[i,1],'rx', markersize=12)
    # for i in range(0,len(points),100):
    #     plt.plot([p1[i,0],p1[i,0]+v1[i,0]],[p1[i,1],p1[i,1]+v1[i,1]],'r')
    #     plt.plot([p1[i,0]],[p1[i,1]],'rx')
    #     plt.plot([p1[i,0]+v1[i,0]],[p1[i,1]+v1[i,1]],'ro')


    #     plt.plot([p3[i,0],p3[i,0]+v2[i,0]],[p3[i,1],p3[i,1]+v2[i,1]],'m')
    #     plt.plot([p3[i,0]],[p3[i,1]],'mx')
    #     plt.plot([p3[i,0]+v2[i,0]],[p3[i,1]+v2[i,1]],'mo')
    #     plt.plot([0,points[i,0]],[0,points[i,1]],'g-x')
    #     plt.text(points[i,0],points[i,1],f"{i}")
    # plt.subplot(222)
    # for i in range(0,len(a1),100):
    #     plt.text(i,a1[i]/np.pi*180,f"{i}")
    #     plt.plot(i,a1[i]/np.pi*180,'rx')
    #     plt.plot(i,a2[i]/np.pi*180,'mx')
    # plt.plot(a1/np.pi*180,'r')
    # plt.plot(a2/np.pi*180,'m')
    # plt.subplot(223)
    # plt.plot(error)
    # for i in idx:
    #     plt.plot(i,error[i],'rx')
    # plt.show()
# @jit(nopython=True,cache=True)
def get_edgeType(normalized_side):
    edgeType = np.zeros((4,))
    # plt.figure()
    # for s in normalized_side:
    #     plt.plot(s[:,0,0],s[:,0,1])
    #     plt.plot()

    for s,i in zip(normalized_side,range(4)):
        p = s[:,0,:].T
        r = s[:,0,0]**2+s[:,0,1]**2
        probeIdx = np.argmax(distLineSegToPoint(p[:,0],p[:,-1],p))
        rm = (r[0]+r[-1])/2
        edgeType[i] = np.sign((r[probeIdx]-rm))
    return edgeType

def distLineSegToPoint(p1,p2,p3):
    """ Distance of point p3  to a line defined by p1 and p2"""
    return np.abs(np.cross((p2-p1).T,(p3-p1[:,None]).T)/np.linalg.norm(p2-p1))

@jit(nopython=True,cache=True)
def get_sides(points, edgeIdx):
    sides = []
    lenEdgeIdx = len(edgeIdx)
    for i in range(lenEdgeIdx):
        if edgeIdx[i]<edgeIdx[(i+1)%lenEdgeIdx]:
            sides.append(points[edgeIdx[i]:edgeIdx[(i+1)%lenEdgeIdx],:,:])
        else:
            sides.append(np.concatenate((points[edgeIdx[i]:,:,:],points[:edgeIdx[(i+1)%lenEdgeIdx],:,:])))
    return sides

@jit(nopython=True,cache=True)
def corner_angle(v1,v2,vo):
    a1 = np.arctan2(v1[:,1],v1[:,0])
    # a1 = np.where(a1<0 , 2*np.pi+a1, a1)
    a2 = np.arctan2(v2[:,1],v2[:,0]) 
    # a2 = np.where(a2<0 , 2*np.pi+a2, a2)
    ao = np.arctan2(vo[:,1],vo[:,0]) 
    # ao = np.where(ao<0 , 2*np.pi+ao, ao)
    return ( (a1-ao) + np.pi) % (2 * np.pi ) - np.pi,( (ao-a2) + np.pi) % (2 * np.pi ) - np.pi

@jit(nopython=True,cache=True)
def angle_between(v1,v2,vo):
    v1_u = (v1 / np.linalg.norm(v1,axis=0))
    v2_u = (v2 / np.linalg.norm(v2,axis=0))
    return np.arccos(np.clip(v1_u[:,0]*v2_u[:,0]+v1_u[:,1]*v2_u[:,1], -1.0, 1.0))






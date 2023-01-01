from calc.SinglePiece import SinglePiece
import matplotlib.pyplot as plt
import numpy as np
import cv2


class PlotPieces():
    def __init__(self):
        self.pieces_map=[]

    def plot_all(self, piecesList:list[SinglePiece]):
        d = 700
        g = 5
        for p,idx in zip(piecesList,range(len(piecesList))):
            x = (idx/g)*d
            y = (idx%g)*d
            plt.plot(p.normalized_points[:,0,0]+x,p.normalized_points[:,0,1]+y)
            plt.plot(p.normalized_points[30,0,0]+x,p.normalized_points[30,0,1]+y,'bx')
            plt.plot(p.normalized_points[-30,0,0]+x,p.normalized_points[-30,0,1]+y,'rx')
            plt.text(x,y,f"{p.pictureName}\n{p.idx}",size=5)

    def plot_piece( self,src:SinglePiece ):
        colors = ["r","b"]
        for p,t in zip(src.normalized_sides,src.edgeType):
            plt.plot(p[:,0,0],p[:,0,1],colors[int((t+1)/2)])

        plt.text(np.mean(src.normalized_points[:,0,0]),np.mean(src.normalized_points[:,0,1]),f"{src.pictureName}\n{src.idx}",size=8)


    def transform_and_plot(self,transform,piece:SinglePiece, pieceIdx=0):
        sideIdx= [np.mean(s,axis=0,keepdims=True)/2 for s in piece.normalized_sides]
        sidePoints = [cv2.transform(s, transform[:2,:]) for s in sideIdx]
        tmp = cv2.transform(piece.normalized_points, transform[:2,:])
        plt.plot(tmp[:,0,0],tmp[:,0,1])
        for i in range(4):
            plt.text(sidePoints[i][0,0,0],sidePoints[i][0,0,1],f"{pieceIdx*4+i}",size=8)
        plt.text(np.mean(tmp[:,0,0]),np.mean(tmp[:,0,1]),f"{piece.pictureName}\n{piece.idx}",size=8)

    def rotate_and_plot(self,rotMat,rotHist,piece:SinglePiece):
        Tr = np.array([[1,0,0],[0,1,0],[0,0,1]])
        for idx in reversed(rotHist):
            Tr_ = rotMat[idx[0],idx[1],:,:]
            # print(rotMat[idx[0],idx[1],:,:])
            Tr = (np.matrix(Tr_)*np.matrix(Tr)).A

        tmp = cv2.transform(piece.normalized_points, Tr[:2,:])

        plt.plot(tmp[:,0,0],tmp[:,0,1])
        plt.text(np.mean(tmp[:,0,0]),np.mean(tmp[:,0,1]),f"{piece.pictureName}\n{piece.idx}",size=8)



    def plot_next_piece( self,src:SinglePiece, trg:SinglePiece, srcSide:int, trgSide:int):
        
        ss = src.normalized_sides[srcSide]
        ts = trg.normalized_sides[trgSide]
        # plt.figure("sides")
        # plt.plot(src.normalized_points[:,0,0],src.normalized_points[:,0,1],'g')
        # plt.plot(trg.normalized_points[:,0,0],trg.normalized_points[:,0,1],'m')
        # plt.plot(ss[:,0,0],ss[:,0,1],'b')
        # plt.plot(ts[:,0,0],ts[:,0,1],'r')
        # find angle of line 1

        # find rotation angle
        a1 = np.arctan2(ss[-1,0,1]-ss[0,0,1], ss[-1,0,0]-ss[0,0,0])
        a2 = np.arctan2(ts[-1,0,1]-ts[0,0,1], ts[-1,0,0]-ts[0,0,0])+np.pi # add pi
        da = a1-a2 # delta angle
        R = np.array(((np.cos(da), -np.sin(da)), (np.sin(da), np.cos(da))))
        T = ((ss[-1,0,:]+ss[0,0,:])/2-(ts[-1,0,:]+ts[0,0,:])/2)[:,None]
        # rotate whole piece
        trgRot = np.dot(R,trg.normalized_points[:,0,:].T).T
        T = ((ss[-1,0,:]+ss[0,0,:])/2-(trgRot[trg.edgeIdx[trgSide],:]+trgRot[trg.edgeIdx[(trgSide+1)%4],:])/2)[None,:]
        trgRotTrans = trgRot + T


        # plt.figure("rotated")
        # plt.plot(src.normalized_points[:,0,0],src.normalized_points[:,0,1],'b')
        plt.plot(trgRotTrans[:,0],trgRotTrans[:,1],'r')


    def _add_to_pieces_map(self,piece:SinglePiece):
        pass
from cmath import pi
from calc.SinglePiece import SinglePiece
import pickle
import os
import cv2
import matplotlib.pyplot as plt

class DataManager():

    def __init__(self):
        pass



    def generatePieces(self,path:str, minArea = 50000, maxArea = 250000) -> list[SinglePiece]:
        imPath = []
        imName = []
        piecesList = []
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path,f)):
                imPath.append(path)
                imName.append(f)

        for imp,imn in zip(imPath,imName):
            im = cv2.imread(os.path.join(imp,imn))

            gray = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)
            _,binary = cv2.threshold(cv2.GaussianBlur(gray,(15,15),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            tmpPieces = []
            for c in contours:
                tmpPieces.append(SinglePiece(c,minArea=minArea,maxArea=maxArea, pictureSrcPath=imp,pictureName=imn))
            
            tmpPieces = [p for p in tmpPieces if p.valid]
            tmpPieces.sort(key=lambda x: (x.center[0][0], x.center[0][1]))
            for p,idx in zip(tmpPieces,range(len(tmpPieces))):
                p.setIdx(idx)
            piecesList.extend(tmpPieces)
            print(f"{len(tmpPieces)} loaded from {imn}")
            
        return piecesList

    def loadPieces(self,path:str) -> list[SinglePiece]:
        if os.path.exists(path):
            with open(path,"rb") as f:
                obj = pickle.load(f)
        return obj

    def storePieces(self,path:str,pieces:list[SinglePiece]):
        with open(path,"wb") as f:
            pickle.dump(pieces,f)
        print(f"{len(pieces)} pieces stored at {path}")

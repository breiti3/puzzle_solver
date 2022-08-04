from __future__ import annotations
import numpy as np



class SinglePiece():

    def __init__(self, points:np.array):
        self.points = points
        self.normalized_points = (points - np.mean(points,axis=1)).astype(np.float32)


    def compare(self, piece:SinglePiece) -> float:
        # compare to oneself and produce an error value
        pass

    

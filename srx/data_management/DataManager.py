from calc.SinglePiece import SinglePiece
import pickle
import os



class DataManager():

    def __init__(self):
        pass




    def load(self,path:str) -> list[SinglePiece]:
        if os.path.exists(path):
            with open(path,"rb") as f:
                out = pickle.load(path)
            return out
        else:
            print("Nothing to load")

    def store(self,path:str,pieces:list[SinglePiece]):
        if os.path.exists(path):
            with open(path,"wb") as f:
                pickle.dump(pieces,f)
            print(f"{len(pieces)} pieces stored at {path}")
        else:
            print("Nothing to load")
import numpy as np



PIECE_IDX_IDX = 0
PIECE_ROT_IDX = 1



class ClusterManager():

    def __init__(self,piece_idx,matrix_size = 100):
        self._pieceMatrix = np.zeros(shape=(matrix_size,matrix_size,2))
        self._pieceMatrix[:] = np.nan

        self.number_of_pieces = 1
        self.piece_idx_list = [piece_idx]

        idx = int(np.floor(matrix_size/2))
        self._pieceMatrix[idx,idx,PIECE_IDX_IDX] = piece_idx
        self._pieceMatrix[idx,idx,PIECE_ROT_IDX] = 0

    def checkForPiece(self,piece_idx):
        boolArray = piece_idx == self._pieceMatrix[:,:,PIECE_IDX_IDX]
        return np.any(boolArray),np.unravel_index(np.argmax(boolArray, axis=None), boolArray.shape)


    def addPiece(self,dst_piece_idx,dst_side_idx,src_piece_idx,src_side_idx):
        # dst_piece_side 0,1,2,3
        # src_piece_side 0,1,2,3
        # source gets added
        # destination should already be there

        # choosen mapping for the sides (x is the piece)
        #       0
        #   3   x   1
        #       2

        # rot 0: no rotation
        # rot 1: piece like this:
        #       3
        #   2   x   0
        #       1
        # rot 2: piece like this:
        #       2
        #   1   x   3
        #       0
        # rot 3: piece like this:
        #       1
        #   0   x   2
        #       3

        exists,idx = self.checkForPiece(dst_piece_idx)
        if not exists:
            return
        dst_side = (dst_side_idx+int(self._pieceMatrix[idx[0],idx[1],PIECE_ROT_IDX]))%4
        # calculate new idx
        newIdx = idx + np.array([-np.cos(dst_side*np.pi/2).astype(int),np.sin(dst_side*np.pi/2).astype(int)])
        if not np.isnan(self._pieceMatrix[newIdx[0],newIdx[1],PIECE_IDX_IDX]):
            print(f"Cluster Error, at location {newIdx} is already a piece")
        # put piece at new location
        self._pieceMatrix[newIdx[0],newIdx[1],PIECE_IDX_IDX] = src_piece_idx
        # turn piece correct amount
        self._pieceMatrix[newIdx[0],newIdx[1],PIECE_ROT_IDX] = (2-src_side_idx+dst_side)%4

        self.number_of_pieces += 1
        self.piece_idx_list.append(src_piece_idx)


        

    def addCluster(self,cluster,dst_piece_idx,dst_side_idx,src_piece_idx,src_side_idx):
        pieces_map = cluster.get_pieces_map(src_piece_idx)
        self.addPiece(dst_piece_idx=dst_piece_idx,dst_side_idx=dst_side_idx,src_piece_idx=src_piece_idx,src_side_idx=src_side_idx)
        for p in pieces_map:
            self.addPiece(dst_piece_idx=p[0],dst_side_idx=p[1],src_piece_idx=p[2],src_side_idx=p[3])

    def generate_transformation_info(self,transformMat):
        # start at center
        idx = int(self._pieceMatrix.shape[0]/2)
        
        visited = np.full((self._pieceMatrix.shape[0],self._pieceMatrix.shape[1]), False)
        transformHist = [np.array([[1,0,0],[0,1,0],[0,0,1]])]
        return self._generate_transformation_info(transformMat,transformHist,np.array([idx,idx]),visited)


    def _generate_transformation_info(self,transformMat,transformHist,idx,visited):
        visited[idx[0],idx[1]] = True
        tmp_idx = np.copy(idx)
        out = {}
        for i,j,nn in zip([-1,0,1,0],[0,1,0,-1],range(4)):
            tmp_idx[0] = idx[0]+i
            tmp_idx[1] = idx[1]+j
            # is there a piece
            if not np.isnan(self._pieceMatrix[tmp_idx[0],tmp_idx[1],PIECE_IDX_IDX]):  
                # calculate pieces
                dst_piece = int(self._pieceMatrix[idx[0],idx[1],PIECE_IDX_IDX])
                dst_side = int((nn-self._pieceMatrix[idx[0],idx[1],PIECE_ROT_IDX])%4)
                src_piece = int(self._pieceMatrix[tmp_idx[0],tmp_idx[1],PIECE_IDX_IDX])
                src_side = int((nn+2-self._pieceMatrix[tmp_idx[0],tmp_idx[1],PIECE_ROT_IDX])%4)
                # is there transformation information?
                transInfo = transformMat[dst_piece*4+dst_side,src_piece*4+src_side]
                if transInfo.max() != transInfo.min():
                    # was this already visited?
                    if not visited[tmp_idx[0],tmp_idx[1]]:
                        transformHist.append(transInfo)
                        out.update(self._generate_transformation_info(transformMat,transformHist,tmp_idx,visited))
        
        Tr = np.array([[1,0,0],[0,1,0],[0,0,1]])
        for Tr_ in reversed(transformHist):
            Tr = (np.matrix(Tr_)*np.matrix(Tr)).A
        out[int(self._pieceMatrix[idx[0],idx[1],PIECE_IDX_IDX])] = Tr
        transformHist.pop()
        return out



    def get_pieces_map(self,piece_idx):
        exists,idx = self.checkForPiece(piece_idx)
        if not exists:
            return
        visited = np.full((self._pieceMatrix.shape[0],self._pieceMatrix.shape[1]), False)
        return self._get_pieces_map(idx,visited)

    
    def _get_pieces_map(self,idx,visited):
        visited[idx[0],idx[1]] = True
        tmp_idx = np.copy(idx)
        out = []
        for i,j,nn in zip([-1,0,1,0],[0,1,0,-1],range(4)):
            tmp_idx[0] = idx[0]+i
            tmp_idx[1] = idx[1]+j
            # is there a piece
            if not np.isnan(self._pieceMatrix[tmp_idx[0],tmp_idx[1],PIECE_IDX_IDX]):
                # was this already visited?
                if not visited[tmp_idx[0],tmp_idx[1]]:
                    dst_piece = int(self._pieceMatrix[idx[0],idx[1],PIECE_IDX_IDX])
                    dst_side = int((nn-self._pieceMatrix[idx[0],idx[1],PIECE_ROT_IDX])%4)
                    src_piece = int(self._pieceMatrix[tmp_idx[0],tmp_idx[1],PIECE_IDX_IDX])
                    src_side = int((nn+2-self._pieceMatrix[tmp_idx[0],tmp_idx[1],PIECE_ROT_IDX])%4)
                    out.append([dst_piece,dst_side,src_piece,src_side])
                    out.extend(self._get_pieces_map(tmp_idx,visited))
        return out


        
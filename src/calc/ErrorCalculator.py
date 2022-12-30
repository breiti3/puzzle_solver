from calc.SinglePiece import SinglePiece
import numpy as np
from calc.PieceCompare import box_compare, type_compare, icp_compare
import time





def calc_error_and_rot_mat(pieces:list[SinglePiece]):
    error = np.zeros((len(pieces)*4,len(pieces)*4))
    error[:] = np.inf
    transformMat = np.zeros((len(pieces)*4,len(pieces)*4,3,3))
    print(f"Start calculating error matrix ...")
    s = time.monotonic()
    startTime = s
    nIterations = (len(pieces)**2-len(pieces))/2
    iterCnt=0
    for i in range(len(pieces)):
        for j in range(i+1,len(pieces)):
            iterCnt += 1
            if time.monotonic()-s > 5:
                s = time.monotonic()
                currentPercentage = 100/nIterations*iterCnt
                speed = (currentPercentage)/(time.monotonic()-startTime) # percentage per time
                print(f"{currentPercentage:.2f}% done. Remaining time estimation: {((100-currentPercentage)/speed)/60:.2f}min")
            error[4*i:4*i+4,4*j:4*j+4] = 0
            error[4*i:4*i+4,4*j:4*j+4] = type_compare(pieces[i],pieces[j])
            error[4*i:4*i+4,4*j:4*j+4] = box_compare(pieces[i],pieces[j],maxPixelDelta=18*2,matrix=error[4*i:4*i+4,4*j:4*j+4])#18 pixel is about 1mm
            error[4*i:4*i+4,4*j:4*j+4],transformMat[4*i:4*i+4,4*j:4*j+4,:,:] = icp_compare(pieces[i],pieces[j], maxError = 20,matrix=error[4*i:4*i+4,4*j:4*j+4])#18 pixel is about 1mm

    i_lower = np.tril_indices(error.shape[0], -1)
    error[i_lower] = error.T[i_lower]
    tmp = np.transpose(transformMat,(1,0,2,3))
    for i1,i2 in zip(i_lower[0],i_lower[1]):
        if tmp[i1,i2,2,2] == 1:
            transformMat[i1,i2,:,:] = np.linalg.inv(tmp[i1,i2,:,:])
            transformMat[i1,i2,2,:] = np.array([0,0,1])

    return error,transformMat


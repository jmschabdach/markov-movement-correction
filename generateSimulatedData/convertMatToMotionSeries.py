from __future__ import print_function
import numpy as np
import random
import argparse
import scipy.io as spio
import os
import math

"""
Convert a series of .mat transform files into a motion series file in the 
format used by FSL POSSUM.
"""

class BadDirException(Exception):
    def __init__(self, value=None, *args, **kwargs):
        self.parameter = value
        print(value)

    def __str__(self):
        return repr(self.parameter)


def convertMatToRow(matFn):
    """
    Read in a .mat file containing a transformation matrix and convert the
    transformation information into a single row. Return the row as a string.

    Input format:
    AffineTransform_double_3_3: [1x12] array of 
        [ a b c d e f g h i m n p ]
      that should reformat to a rotation-translation matrix:
        [ a  b  c  m ]
        [ d  e  f  n ] 
        [ g  h  i  p ]
        [ 0  0  0  1 ]
    fixed: [1x3] array indicating the center of the transformation

    Return format:
        time, Tx, Ty, Tz, Rx, Ry, Rz
      where T = translation and R = rotation
      Translation is in meters
      Rotation is in radians

    Inputs:
    - matFn: the file path to the .mat file
    - timepoint: the timepoint at which the motion will occur in the 
                 simulated data

    Returns:
    - row: string representing the motion at the given timepoint

    *Matrix conversion calculations determined using the following sources:
    - https://github.com/ANTsX/ANTs/wiki/ITK-affine-transform-conversion
    - https://github.com/hinerm/ITK/blob/master/Modules/Core/Transform/include/itkMatrixOffsetTransformBase.hxx#L724-L745
    """
    # Read in the .mat file
    data = spio.loadmat(matFn)
    # Extract the parameters from the matrix
    mat = data['AffineTransform_double_3_3'][:9]
    mat = np.reshape(mat, (3,3))
    translation = data['AffineTransform_double_3_3'][9:].flatten()
    center = data['fixed'].flatten()
    # Translating the code from ITK
    offset = [0.0, 0.0, 0.0]
    for i in xrange(3): # 3 output variables
        offset[i] = translation[i]+center[i]
        for j in xrange(3): # 3 variables in the rotation matrix
            offset[i] -= mat[i][j]*center[j]

    if not np.asarray(offset).all() == translation.all():
        print("Note: calculated offset is not equal to the translation.")

    # Extract the translation and rotation parameters
    # already have the translation, but convert from mm to m
    
    # scaling
    scaling = [0.0, 0.0, 0.0]
    for i in xrange(len(mat)):
        scaling[i] = np.linalg.norm([mat[0][i], mat[1][i], mat[2][i]])
    # if the scaling is not approximately 1
    for i in xrange(len(mat)):
        if not np.isclose(scaling[i], 1):
            # scale the appropriate column in the rotation matrix
            for j in xrange(len(mat[0])):
                mat[j][i] = mat[j][i]/scaling[i]

    # Extract the angles from the rotation matrix
    rotationString = extractAnglesFromRotationMatrix(mat)
    # Convert the translation and rotation parameters into a string
    translationString = '{:16.7E}'.format(translation[0]/1000)+'{:16.7E}'.format(translation[1]/1000)+'{:16.7E}'.format(translation[2]/1000)
    # Return the row string
    rowString = translationString+rotationString
    return rowString


def extractAnglesFromRotationMatrix(matrix):
    """
    Given a rotation matrix, extract the x, y, and z angles and return them
    as a string (to be used in FSL POSSUM)

    Input:
    - matrix: a 3x3 unscaled rotation matrix

    Returns:
    - rotationString: a string representing the rotation

    Code sourced from: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    # Initialize return string
    rotationString = ""

    # Check if the matrix is a valid rotation matrix
    matrixT = np.transpose(matrix)
    shouldBeIdentity = np.dot(matrixT, matrix)
    I = np.identity(3, dtype = matrix.dtype)
    n = np.linalg.norm(I-shouldBeIdentity)
    
    if n < 1e-6:
        # check if the matrix is singular
        sy = math.sqrt(matrix[0,0]**2 + matrix[1,0]**2)
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(matrix[2,1], matrix[2,2])
            y = math.atan2(-matrix[2,0], sy)
            z = math.atan2(matrix[1,0], matrix[0,0])
        else:
            x = math.atan2(-matrix[1,2], matrix[1,1])
            y = math.atan2(-matrix[2,0], sy)
            z = 0
        # Since math.atan2 returns angles in radians, no conversion needed
        rotationString = '{:16.7E}'.format(x)+'{:16.7E}'.format(y)+'{:16.7E}'.format(z)
    return rotationString


def main():
    # Set up the argparser
    parser = argparse.ArgumentParser(description='Convert a collection of .mat files into a motion series file that can be used with FSL POSSUM.')
    parser.add_argument('-d', '--matdirectory', type=str, help='Full path to the directory containing the .mat files')
    parser.add_argument('-o', '--outputfile', type=str, help='Full path + name of the file where the motion sequence should be saved')
    
    args = parser.parse_args()
    # Parse the directory containing the .mat files
    matDir = args.matdirectory
    # Check that the directory actually exists
    if not os.path.exists(matDir):
        raise BadDirException("The directory "+matDir+" does not exist.")

    matFns = sorted(os.listdir(matDir))
    # Parse the filename where the motion series will be saved
    outFn = args.outputfile
   
    timepoint = 0.0
    # With the given save file open
    with open(outFn, 'w') as f:
        # For each .mat file
        row = '{:16.7E}'.format(timepoint)+'{:16.7E}'.format(0.0)+'{:16.7E}'.format(0.0)+'{:16.7E}'.format(0.0)+'{:16.7E}'.format(0.0)+'{:16.7E}'.format(0.0)+'{:16.7E}'.format(0.0)+'\n'
        f.write(row)
        timepoint = timepoint+2.45
        for matFn in matFns:
            # Convert the .mat file to a row of transformation parameters
            row = '{:16.7E}'.format(timepoint-.2)+convertMatToRow(matDir+matFn)+'\n'
            f.write(row)
            timepoint = timepoint+2.45

    # Print "completion" message
    print("The .mat files have been converted to a motion sequence.")

if __name__ == "__main__":
    main()

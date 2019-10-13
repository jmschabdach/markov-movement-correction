from __future__ import print_function
import numpy as np

# Load custom library
from boldli import ImageManipulatingLibrary as mil

# For loading/saving the images
from nipy.core.api import Image
from nipy import load_image, save_image
from nipype.interfaces import dcmstack

# For generating the mask
import SimpleITK as sitk

# For calculating mutual information
from scipy.stats import entropy

# For command line arguments
import argparse

# For saving outputs
import os


##
# Generate a 3D volumetric brain mask from the chosen image volume
#
# @param vol The image volume
# 
# @returns mask The image mask
def generateMask(vol):
    # Perform Otsu thresholding
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    mask = otsu.Execute(vol)

    return mask

##
# Use Otsu thresholding to convert a sequence into binary images
#
# @param seq The image sequence
#
# @returns binarized The binarized image sequence
def binarizeSequence(sequence):

    binarized = []

    for i in range(sequence.shape[-1]):
        # Isolate the volume
        vol = mil.isolateVolume(sequence, volNum=i)
        # Convert the volume from an array to an image
        volImg = sitk.GetImageFromArray(vol)
        # Get the mask of the volume
        mask = generateMask(volImg)
        # Get the binarized image as an array
        binarized.append(sitk.GetArrayFromImage(mask))

    return binarized

##
# Calculate the Dice coefficient between a pair of image volumes
#
# @param img1 One image volume
# @param img2 Second image volume
#
# @returns coeff The Dice coefficient describing the overlap between these images
def calculateDice(img1, img2):
    
    # Convert the images from 3D arrays to 1D arrays
    arr1 = img1.flatten()
    arr2 = img2.flatten()

    # Count the number of elements that are 1 in both images
    overlap = float((arr1 == arr2).sum())

    # The dice coefficient is 2*h/(a+b) where
    #    h is the overlap between the two images
    #    a is the length of image 1
    #    b is the length of image 2
    # Since len(a) == len(b), we simplify to h/a
    coeff = 2.0*overlap/(float(len(arr1))+float(len(arr2)))

    return coeff

##
# Calculate the mutual information between a pair of image volumes
#
# @param img1 One image volume
# @param img2 Second image volume
#
# @returns mi The mutual information of the pair of image volumes
def calculateMI(img1, img2):
    # MI(img1, img2) = H(img1) + H(img2) - H(img1, img2)
    # where H(.) is the entropy

    # Convert the images from 3D arrays to 1D arrays
    arr1 = img1.flatten()
    arr2 = img2.flatten()
 
    # Calculate the marginal and joint histograms of the images (empiracal probabilities)
    p_x, edgesX = np.histogram(arr1)
    p_y, edgesY = np.histogram(arr2)
    p_xy, x_edges, y_edges = np.histogram2d(arr1, arr2)

    # Calculate the entropy of the marginal and joint probabilities
    h_xy = entropy(p_xy.flatten())
    h_x = entropy(p_x)
    h_y = entropy(p_y)

    # Calculate the mutual information
    mi = h_x + h_y - h_xy

    return mi



def main():
    # Future: add arguments for inFn, outFn, and volNum
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inFn", type=str, help="Input image filename")

    # Parse the arguments
    args = parser.parse_args()

    # Specify filepaths
    #inFn = "/home/jenna/Documents/Pitt/CHP-PIRC/markov-movement-correction/data/test_image/BOLD.nii.gz"
    inFn = args.inFn

    # Check to make sure the input file exists
    if not os.path.exists(inFn):
        print("File doesn't exist:", inFn)
        exit(-1)

    # Load the original image sequence
    sequence, coordinates = mil.loadBOLD(inFn)

    # Binarize the sequence
    binarizedArray = binarizeSequence(sequence)

    # Convert the binarized image back to an image
    #maskStack = mil.convertArrayToImage(binarizedArray, coordinates)
    #mil.saveBOLD(maskStack, "/home/jenna/Documents/Pitt/CHP-PIRC/markov-movement-correction/data/test_image/binarized_brain.nii.gz")

    # Create the matrices for the Dice coefficients and the MI
    matDice = np.zeros((sequence.shape[-1], sequence.shape[-1]))
    matMI = np.zeros((sequence.shape[-1], sequence.shape[-1]))

    # Iterate through the volumes in the sequence
    for i in range(len(matDice)): # row
        for j in range(len(matDice[0])): # col
            matDice[i][j] = calculateDice(binarizedArray[i], binarizedArray[j])
            matMI[i][j] = calculateMI(mil.isolateVolume(sequence, volNum=i),
                                      mil.isolateVolume(sequence, volNum=j))
        print("Finished calculating for row", i)

    # Save the matrices as .csv files in the image's metrics subdirectory
    outDir = os.path.join(os.path.dirname(inFn), "metrics")
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    fnDice = os.path.join(outDir, split(os.path.basename(inFn), ".")[0]+"_dice_mat.csv")
    fnMI = os.path.join(outDir, split(os.path.basename(inFn), ".")[0]+"_mi_mat.csv")
    np.savetxt(fnDice, matDice, delimiter=",")
    np.savetxt(fnMI, matMI, delimiter=",")

    # Announce completion to the user
    print("Finished calculating Dice and MI")


if __name__ == "__main__":
    main()

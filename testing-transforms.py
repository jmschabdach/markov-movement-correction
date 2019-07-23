from __future__ import print_function
import numpy as np
import getpass
import os
import argparse
import time
import shutil
import sys
import time

# for loading/saving the images
from nipy.core.api import Image
from nipy import load_image, save_image

# for the registration
from nipype.interfaces.ants import Registration, ApplyTransforms
from nipype.algorithms.metrics import Similarity

# for saving the registered file
from nipype.interfaces import dcmstack

# threading
import threading

def alignCompartments(fixedImg, movingImgs, transform):
    """
    Given a precalculated linking transform and a fixed image (required by ANTS, not 
    sure why), align each image in the movingImgs list to the fixed image.

    Inputs:
    - fixedImg: the path to the fixed image
    - movingImgs: a list of paths to the moving images
    - transform: either a list of paths or a single path to a transform file

    Returns:
    - None

    Effects:
    - Overwrites the specified images with a more aligned version of the same images

    *** Note: this version assumes the same fixed image. Could also be implemented 
              so that the fixed image is the previous moving image.
    """
    # for each image
    for m in movingImgs:
        # set up the transform application
        at = ApplyTransforms()
        at.inputs.input_image = m
        at.inputs.reference_image = fixedImg
        at.inputs.output_image = m
        at.inputs.transforms = transform
        at.inputs.interpolation = 'NearestNeighbor'
        at.inputs.invert_transform_flags =[True]
        # run the transform application
        at.run()

baseDir = '/home/jenna/Research/CHP-PIRC/markov-movement-correction/0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001/testing/'

# make the output directory
outputDir = baseDir + 'stacking-hmm/'
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

# tmpDir is not a global variable
tmpDir = baseDir + 'tmp/'
if not os.path.exists(tmpDir):
    os.mkdir(tmpDir)

if not os.path.exists(tmpDir+"linkingTransforms/"):
    os.mkdir(tmpDir+"linkingTransforms/")

numImgs = 150
origTimepoints = [baseDir+'timepoints/'+str(i).zfill(3)+".nii.gz" for i in xrange(150)]

# Step 1: Divide the time series into compartments
# flip the timepoints (goal: align with original timepoint 0)
timepointFns = list(reversed([outputDir+str(i).zfill(3)+".nii.gz" for i in xrange(150)]))
numCompartments = 6
imgsPerCompartment = int(np.ceil(len(timepointFns)/float(numCompartments)))
# make the list of lists
hmmCompartments = [timepointFns[i*imgsPerCompartment:(i+1)*imgsPerCompartment] for i in xrange(numCompartments-1)]
hmmCompartments.append(timepointFns[imgsPerCompartment*(numCompartments-1):])

transformPrefix = tmpDir+"prealignTransforms/stacking-hmm_"
compartmentTransformFns = [transformPrefix+str(i)+'_1Affine.mat' for i in xrange(numCompartments)]
linkingTransFns = [tmpDir+"linkingTransforms/compartment"+str(i)+"_compartment"+str(i+1)+'_0GenericAffine.mat' for i in xrange(numCompartments-1)]

print(len(compartmentTransformFns))
print(len(linkingTransFns))

# Step 4: apply linking transform to each compartment
alignedFns = []
refImg = origTimepoints[0] # reference image required, only for metadata
for i in xrange(len(hmmCompartments)-1):    # 1 less linking transform
    # add the current compartment to the list
    alignedFns.extend(hmmCompartments[i])
    # change the alignment from the first image in the newly added
    #  compartment to the last image in the compartment
    alignCompartments(refImg, alignedFns[:-1], compartmentTransformFns[i])
    # now link all the images to the first image in the compartment
    #  that hasn't been added yet (the next one)
    alignCompartments(refImg, alignedFns, linkingTransFns[i])

# now we're in the last compartment
alignedFns.extend(hmmCompartments[-1])
# need to apply the final compartment transform function (this needs a better name)

# # **** IMPORTANT: when perfected, remove this step
# # copy over the hmm registered images to a new directory
spareDir = baseDir+"stackingHmmNoFinalTransform/"
if not os.path.exists(spareDir):
    os.mkdir(spareDir)
for compartment in hmmCompartments:
    for image in compartment:
        shutil.copy2(image, spareDir)
        print(image)

# this is the line that takes it from ok to horrible
alignCompartments(refImg, alignedFns[:-1], compartmentTransformFns[-1])
print(compartmentTransformFns[-1])
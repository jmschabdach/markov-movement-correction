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

"""
Check to make sure the images and results processed on both DBMI and PIRC computers are the same.
"""

# Examine all subjects
baseDir = '/home/jenna/Documents/Pitt/CHP-PIRC/markov-movement-correction/data/'
checkDir = baseDir + "JennaS_Bolds_To_Check/"
expDir = baseDir + "clean-data/LinearControls/"

checkImgs = [checkDir+'00114_BOLD_06-23-2014.nii.gz',
             checkDir+"0114_06-28-2014.nii.gz",
             checkDir+"0116_08-29-2014.nii.gz",
             checkDir+"0116_09-06-2014.nii.gz",
             checkDir+"0473_04-16-2013.nii.gz",
             checkDir+"0473_5-29-2013.nii.gz"]

expImgs = [expDir+'0114-2/BOLD.nii',
           expDir+'0114-2/BOLD.nii',
           expDir+'0116-2/BOLD.nii',
           expDir+'0116-2/BOLD.nii',
           expDir+'0473_TC_070_02a/BOLD.nii',
           expDir+'0473_TC_070_02a/BOLD.nii']

for checkImgFn, expImgFn in zip(checkImgs, expImgs):

    # Check the original BOLD images
    checkImg = load_image(checkImgFn)
    expImg = load_image(expImgFn)

    checkData = np.asarray(checkImg.get_data())
    expData = np.asarray(expImg.get_data())

    if np.array_equal(checkData, expData):
        print('Images equal:')
        print(checkImgFn)
        print(expImgFn)
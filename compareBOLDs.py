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

# Examine all subjects
baseDir = '/home/jenna/Research/CHP-PIRC/markov-movement-correction/'
linearDir = baseDir + "LinearControls/"
nonlinearDir = baseDir + "Controls/"

linearDirs = [subjDir for subjDir in os.listdir(linearDir) if os.path.isdir(os.path.join(linearDir, subjDir))]
nonlinearDirs = [subjDir for subjDir in os.listdir(nonlinearDir) if os.path.isdir(os.path.join(nonlinearDir, subjDir))]

print(len(linearDirs))
print(len(nonlinearDirs))

for subjLinearDir, subjNonlinearDir in zip(linearDirs, nonlinearDirs):
    print(subjLinearDir, subjNonlinearDir)

    # Check the original BOLD images
    origLinear = linearDir+subjLinearDir+"/BOLD.nii"
    origNon = nonlinearDir+subjNonlinearDir+"/BOLD.nii"

    linearImg = load_image(origLinear)
    nonlinearImg = load_image(origNon)

    linearData = linearImg.get_data()
    nonlinearData = nonlinearImg.get_data()

    print("Images same:",np.array_equal(linearData, nonlinearData))

    # Check the correlation ratio matrices
    linearMatrixFn = linearDir+subjLinearDir+"/metrics/crossCorrelation-timepoints.csv"
    nonlinearMatrixFn = nonlinearDir+subjNonlinearDir+"/metrics/crossCorrelation-timepoints.csv"

    linearMatrix = np.loadtxt(open(linearMatrixFn, 'r'), delimiter=',')
    nonlinearMatrix = np.loadtxt(open(nonlinearMatrixFn, 'r'), delimiter=',')

    print("Correlation ratio matrices same:",np.array_equal(linearMatrix, nonlinearMatrix))

"""
BASE1='/home/jenna/Research/CHP-PIRC/markov-movement-correction/LinearControls/0068'
TIME1="$BASE1/timepoints/"
TIME_FN1="$BASE1/metrics/crossCorrelation-timepoints.csv"

BASE2='/home/jenna/Research/CHP-PIRC/markov-movement-correction/Controls/0068'
TIME2="$BASE2/timepoints/"
TIME_FN2="$BASE2/metrics/crossCorrelation-timepoints.csv"
bash calculateCorrelationMatrix-2.sh $TIME1 $TIME_FN1 & bash calculateCorrelationMatrix-3.sh $TIME2 $TIME_FN2
 """
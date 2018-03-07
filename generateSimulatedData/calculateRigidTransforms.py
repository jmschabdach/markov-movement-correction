from __future__ import print_function
import numpy as np
import argparse
import os

from nipy.core.api import Image
from nipy import load_image, save_image
from nipype.interfaces.ants import Registration, ApplyTransforms
from nipype.algorithms.metrics import Similarity
from nipype.interfaces import dcmstack

import threading

"""
Given an rs-fMR image, calculate the rigid transformations between each pair ofsequential frames. Save the calculated transformations to the specified directory. If the directory doesn't exist, make it.
"""

class pairwiseRegistrationThread(threading.Thread):
    def __init__(self, threadId, frame1, frame2, transformDir):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.frame1 = frame1
        self.frame2 = frame2
        self.transformDir = transformDir

    def run(self):
        print("Starting the registration for", self.frame1, "and", self.frame2)
        calculateRigidTransforms(self.frame1, self.frame2, self.transformDir)

def expandFrames(imgFn, saveDir):
    """
    Expand a timeseries image into a set of individual frames in the
    specified directory

    Inputs:
    - imgFn: the timeseries image's filename
    - saveDir: the directory in which the frames will be stored

    Returns:
    - frameFns: the list of filenames
    """
    # Load the image
    img = load_image(imgFn)
    coord = img.coordmap
    frameFns = []

    # Make the save directory
    framesDir = saveDir+'/frames/' # need to check for //
    # check for duplicate //
    framesDir = framesDir.replace("//", '/')
    if not os.path.exists(framesDir):
        os.mkdir(framesDir)

    for i in xrange(img.get_data().shape[3]):
        frame = img[:,:,:,i].get_data()[:,:,:,None]
        frameImg = Image(frame, coord)
        outFn = framesDir+str(i).zfill(3)+".nii.gz"
        save_image(frameImg, outFn)
        frameFns.append(outFn)

    return frameFns


def calculateRigidTransforms(frame1, frame2, saveFn):
    """
    Given the pair of images, calculate the rigid transformation from frame2
    to frame1 and save it using the saveFn prefix.

    Inputs:
    - frame1: image at timepoint n
    - frame2: image at timepoint n+1
    - saveFn: the prefix filename where the transform will be saved
    """
    # set up the registration
    reg = Registration()
    reg.inputs.fixed_image = frame1
    reg.inputs.moving_image = frame2
    reg.inputs.output_transform_prefix = saveFn
    reg.inputs.interpolation = 'NearestNeighbor'

    reg.inputs.transforms = ['Rigid']
    reg.inputs.transform_parameters = [(0.1,)]
    reg.inputs.number_of_iterations = [[500, 20]]
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = False
    reg.inputs.collapse_output_transforms = True
    reg.inputs.initialize_transforms_per_stage = False
    reg.inputs.metric = ['MI']
    reg.inputs.metric_weight = [1]
    reg.inputs.radius_or_number_of_bins = [32]
    reg.inputs.sampling_strategy = ['Random']
    reg.inputs.sampling_percentage = [0.05]
    reg.inputs.convergence_threshold = [1.e-2]
    reg.inputs.convergence_window_size = [20]
    reg.inputs.smoothing_sigmas = [[2,1]]
    reg.inputs.sigma_units = ['vox']
    reg.inputs.shrink_factors = [[2,1]]

    reg.inputs.use_estimate_learning_rate_once = [True]
    reg.inputs.use_histogram_matching = [True]
    reg.inputs.output_warped_image = False
    reg.inputs.num_threads = 50

    # run the registration
    reg.run()

def main():
    # Set up the argparser
    parser = argparse.ArgumentParser(description="Calculate the rigid transformation between all pairs of frames in the specified image.")
    parser.add_argument('-i', '--image', type=str, help='Full path to the name of the file to calculate registrations for')
    parser.add_argument('-d', '--savedir', type=str, help='Path of the directory where the registered frames and the transforms will be saved')

    args = parser.parse_args()

    # Parse the image
    imgFn = args.image 
    # Parse the directory
    saveDir = args.savedir
    # check that the directory exists and make it if it doesn't exist
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    
    # Expand the image into a set of frames
    frames = expandFrames(imgFn, saveDir)

    # Make the directory to store the transforms
    transformsDir = saveDir+'/transforms/'
    transformsDir = transformsDir.replace('//', '/')
    if not os.path.exists(transformsDir):
        os.mkdir(transformsDir)

    # Make a list of threads
    threads = []

    for i in xrange(len(frames)-1):
        # make the prefix for the transform
        prefix = transformsDir+'rigidTransform_'+str(i).zfill(3)+"_"+str(i+1).zfill(3)+"_"
        # start a thread to register the images
        t = pairwiseRegistrationThread(i, frames[i], frames[i+1], prefix)
        threads.append(t)
        t.start()
    
    # Make sure the threads are finished running
    for t in threads:
        t.join()
    
    # print a "finished" message
    print("Finished calculating all rigid transformations for image.")

if __name__ == "__main__":
    main()

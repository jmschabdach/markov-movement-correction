from __future__ import print_function
import numpy as np
import getpass
import os
import argparse

# for loading/saving the images
from nipy.core.api import Image
from nipy import load_image, save_image

# for the registration
from nipype.interfaces.ants import Registration

# for saving the registered file
from nipype.interfaces import dcmstack

# threading
import threading
from thread import allocate_lock
import time

# add threading class that specifies the values of each parameter to be tested
#---------------------------------------------------------------------------------
# Threading Class
#---------------------------------------------------------------------------------
class myThread(threading.Thread):
    """
    Implementation of the threading class.
    """
    def __init__(self, threadId, templateFn, timepointFn, outputFn, outputDir, hp):
        # What other properties my threads will need?
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.templateFn = templateFn
        self.timepointFn = timepointFn
        self.outputFn = outputFn
        self.outputDir = outputDir
        self.hyperparameters = hp.copy()

    def run(self):
        print("Starting motion correction for", self.name)
        startTime = time.clock()
        registerToTemplate(self.templateFn, self.timepointFn, self.outputFn, self.outputDir, self.hyperparameters)
        timePassed = time.clock() - startTime
        print("Time passed:", timePassed)
        lock.acquire()
        # need to save hyperparameters and amount of time taken
        saveParams(hyperparameters, timePassed)
        lock.release()
        print("Finished motion correction for", self.name)


# add registration function that takes in the values of each parameter to be tested
def registerToTemplate(fixedImgFn, movingImgFn, outFn, outDir, hyperparams):
    """
    Register 2 images taken at different timepoints.

    Inputs:
    - fixedImgFn: filename of the fixed image (should be the template image)
    - movingImgFn: filename of the moving image (should be the Jn image)
    - outFn: name of the file to write the transformed image to.
    - outDir: path to the tmp directory
    - hyperparameters: hyperparameters to optimize

    Outputs:
    - Currently, nothing. Should return or save the transformation

    Effects:
    - Saves the registered image
    """
    reg = Registration()
    reg.inputs.num_threads = 4
    reg.inputs.number_of_iterations = [[50]]
    reg.inputs.fixed_image = fixedImgFn
    reg.inputs.moving_image = movingImgFn
    reg.inputs.output_transform_prefix = outDir+"output_"
    reg.inputs.transforms = ['SyN']
    reg.inputs.transform_parameters = [(0.25, 3.0, 0.0)]  # should this be fine-tuned as well?
    reg.inputs.number_of_iterations = [[100, 50, 30]]
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = False
    reg.inputs.initialize_transforms_per_stage = False
    reg.inputs.metric = ['CC']
    reg.inputs.metric_weight = [1] # Default (value ignored currently by ANTs)
    reg.inputs.radius_or_number_of_bins = [hyperparams["radius_or_number_of_bins"]]  # grid-searching this 
    reg.inputs.sampling_strategy = [None]
    reg.inputs.sampling_percentage = [None]
    reg.inputs.convergence_threshold = [hyperparams["convergence_threshold"]]  # grid-searching this
    reg.inputs.convergence_window_size = [hyperparams["convergence_window_size"]]  # grid-searching this
    reg.inputs.smoothing_sigmas = [hyperparams["smoothing_sigmas"]]  # grid-searching this
    reg.inputs.sigma_units = ['vox'] * 2
    reg.inputs.shrink_factors = [hyperparams["shrink_factors"]]  # grid-searching this
    reg.inputs.use_estimate_learning_rate_once = [True]
    reg.inputs.use_histogram_matching = [True] # This is the default
    reg.inputs.output_warped_image = outFn

    # if initialize is not None:
    #     reg.inputs.initial_moving_transform = initialize
    #     reg.inputs.invert_initial_moving_transform = False

    # print(reg.cmdline)
    print("Starting registration for",outFn)
    reg.run()
    print("Finished running registration for", outFn)


def saveParams(hyperparams, threadTime, baseDir):
    line = str(hyperparams["radius_or_number_of_bins"])+", "+str(hyperparams["convergence_threshold"])+", "+ str(hyperparams["convergence_window_size"])+", "+str(hyperparams["smoothing_sigmas"])+", "+str(hyperparams["shrink_factors"])+", " + str(threadTime) + "\n"
    f = open(baseDir+'ANTsOpt_hyperparams.csv', 'a')
    f.write(line)
    f.close()
    pass


def main(baseDir):
    # read the images
    imgFn1 = baseDir+ "o_TC_005_T2_T2_Bias_Corrected.nii.gz"
    imgFn2 = baseDir+ "o_TC_010_T2_T2_Bias_Corrected.nii.gz"
    imgFn3 = baseDir+ "TC_050_01a)S08CORT23DSPACE_T2_Bias_Corrected.nii.gz"
    # set up the list of hyperparameter values that can be used 
    #  and the different values to test
    hyperparameters = {
        "radius_or_number_of_bins": [64, 32, 16, 8], #[48, 44, 40, 36, 32, 28, , 8],
        "convergence_threshold": [1., 1.e-2, 1.e-4, 1.e-5, 1.e-7, 1.e-9],
        "convergence_window_size": [5, 10, 20, 30, 40, 50],
        "smoothing_sigmas": [[2,1,0], [5,5,5], [4,4,4], [3,3,3], [2,2,2], [1,1,1], [0,0,0]],
        "shrink_factors": [[3,2,1], [3], [6,6,6], [5,5,5], [4,4,4], [3,3,3], [2,2,2], [1,1,1]]
    }
    defaults = {
        "radius_or_number_of_bins": 32,
        "convergence_threshold": 1.e-9,
        "convergence_window_size": 20,
        "smoothing_sigmas": [2,1,0],
        "shrink_factors": [3,2,1]
    }
    # initialize the parameter info file
    line = "radius_or_number_of_bins, convergence_threshold, convergence_window_size, smoothing_sigmas, shrink_factors, time\n"
    f = open(baseDir+'ANTsOptimal_hyperparams.csv', 'w+')
    f.write(line)
    f.close()

    # set up each registration 
    # threadId = 1
    outDir = baseDir+"tmp/ANTsOptimization/"
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    # lock = allocate_lock()

    for key, optsList in hyperparameters.iteritems():
        for opt in optsList:
            hp = defaults.copy()
            hp[key] = opt
            # make the output filename
            outFn = outDir+str(key)+"_"+str(opt)+".nii.gz"

            # start the thread for those hyperparameters
            # t = myThread(threadId, imgFn1, imgFn2, outFn, outDir, hp)
            # t.start()
            # threadId += 1

            # realized threading was a bad idea - can't get an accurate time-for-registration
            startTime = time.clock()
            registerToTemplate(imgFn1, imgFn2, outFn, outDir, hyperparams)
            totalTime = time.clock() - startTime
            print("Registration completed in", totalTime)
            saveParams(hyperparams, totalTime, baseDir)

if __name__ == "__main__":
    # set the base directory
    # baseDir = '/home/pirc/Desktop/Jenna_dev/markov-movement-correction/'
    # baseDir = '/home/pirc/processing/FETAL_Axial_BOLD_Motion_Processing/markov-movement-correction/'
    baseDir = '/home/jms565/Research/CHP-PIRC/markov-movement-correction/'
    # baseDir = '/home/jenna/Research/CHP-PIRC/markov-movement-correction/'
    main(baseDir)
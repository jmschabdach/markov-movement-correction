from __future__ import print_function
import numpy as np
import getpass
import os
import argparse
import time
import shutil
import sys

# for loading/saving the images
from nipy.core.api import Image
from nipy import load_image, save_image

# for the registration
from nipype.interfaces.ants import Registration

# for saving the registered file
from nipype.interfaces import dcmstack

# threading
import threading

"""
The functions in this file can be used to perform different types of movement correction on a time series of 3D images.
"""

#---------------------------------------------------------------------------------
# Threading Classes
#---------------------------------------------------------------------------------
class motionCorrectionThread(threading.Thread):
    """
    Implementation of the threading class.
    """
    def __init__(self, threadId, name, templateFn, timepointFn, outputFn, outputDir, prealign=False):
        # What other properties my threads will need?
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = name
        self.templateFn = templateFn
        self.timepointFn = timepointFn
        self.outputFn = outputFn
        self.outputDir = outputDir
        self.prealign = prealign

    def run(self):
        print("Starting motion correction for", self.name)
        if not self.prealign:
            registerToTemplate(self.templateFn, self.timepointFn, self.outputFn, self.outputDir)
        else:
            registerToTemplatePrealign(self.templateFn, self.timepointFn, self.outputFn, self.outputDir)
        print("Finished motion correction for", self.name)

class bifurcatedMarkovThread(threading.Thread):
    """
    Implementation of the threading class.

    Purpose: allow for bifurcating markov motion correction. 
    """
    def __init__(self, threadId, threadName, filenames, outputDir, baseDir):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = threadName
        self.fns = filenames
        self.outputDir = outputDir
        self.baseDir = baseDir
        self._return = None

    def run(self):
        # if self._Thread__target is not None:
        print("Starting the bifurcated markov motion correction for", self.name)
        #print("Input files:")
	#print(self.fns)
        outfiles = markovCorrection(self.fns, self.outputDir, self.baseDir, corrId=self.threadId)
        # outfiles = ["ohai", self.name]
        # time.sleep(20/self.threadId)
        print("Finished the bifurcated markov motion correction for", self.name)
        self._return = outfiles

    def join(self):
        threading.Thread.join(self)
        return self._return

#---------------------------------------------------------------------------------
# Motion Correction: Helper Functions
#---------------------------------------------------------------------------------
def expandTimepoints(imgFn, baseDir):
    """
    Expand a time series image into individual images in the tmp folder

    Inputs:
    - imgFn: the time series image's filename
    - outDir: the directory to write files to

    Returns:
    - filenames: list of filenames
    """
    # load the image
    img = load_image(imgFn)
    print(img.get_data().shape)
    coord = img.coordmap

    if not os.path.exists(baseDir+'timepoints/'):
        os.mkdir(baseDir+'timepoints/')
    outDir = baseDir +'timepoints/'

    # pull out the first image timepoint 0
    first = img[:,:,:,0].get_data()[:,:,:,None]
    first_img = Image(first, coord)
    # save the first image as 000, but don't add the name to the list
    save_image(first_img, outDir+str(0).zfill(3)+'.nii.gz')

    # build the list of filenames
    filenames = [outDir+'000.nii.gz']

    # for the remaining images
    for i in xrange(1, img.get_data().shape[3], 1):
        # pull out the image and save it
        tmp = img[:,:,:,i].get_data()[:,:,:,None]
        tmp_img = Image(tmp, coord)
        outFn = str(i).zfill(3)+'.nii.gz'
        save_image(tmp_img, outDir+outFn)
        filenames.append(outDir+outFn)

    return filenames


def registerToTemplate(fixedImgFn, movingImgFn, outFn, outDir, initialize=None, corrId=None):
    """
    Register 2 images taken at different timepoints.

    Inputs:
    - fixedImgFn: filename of the fixed image (should be the template image)
    - movingImgFn: filename of the moving image (should be the Jn image)
    - outFn: name of the file to write the transformed image to.
    - outDir: path to the tmp directory
    - initialize: optional parameter to specify the location of the
                  transformation matrix from the previous registration

    Outputs:
    - Currently, nothing. Should return or save the transformation

    Effects:
    - Saves the registered image
    """
    #print("Output filename:", outFn)

    reg = Registration()
    reg.inputs.fixed_image = fixedImgFn
    reg.inputs.moving_image = movingImgFn
    reg.inputs.output_transform_prefix = outDir+"output_"
    reg.inputs.transforms = ['SyN']
    reg.inputs.transform_parameters = [(0.25, 3.0, 0.0)]
    reg.inputs.number_of_iterations = [[100, 50, 30]]
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = False
    reg.inputs.initialize_transforms_per_stage = False
    reg.inputs.metric = ['CC']
    reg.inputs.metric_weight = [1] # Default (value ignored currently by ANTs)
    reg.inputs.radius_or_number_of_bins = [32]
    reg.inputs.sampling_strategy = [None]
    reg.inputs.sampling_percentage = [None]
    reg.inputs.convergence_threshold = [1.e-9]
    reg.inputs.convergence_window_size = [20]
    reg.inputs.smoothing_sigmas = [[2,1,0]]  # probably should fine-tune these?
    reg.inputs.sigma_units = ['vox'] * 2
    reg.inputs.shrink_factors = [[3,2,1]]  # probably should fine-tune these?
    reg.inputs.use_estimate_learning_rate_once = [True]
    reg.inputs.use_histogram_matching = [True] # This is the default
    reg.inputs.output_warped_image = outFn

    if initialize is not None:
        reg.inputs.initial_moving_transform = initialize
        reg.inputs.invert_initial_moving_transform = False

    if corrId is not None:
        reg.inputs.output_transform_prefix = outDir+"output_"+str(corrId)+"_"

    # print(reg.cmdline)
    print("Starting registration for",outFn)
    reg.run()
    # print(reg.inputs.output_transform_prefix)
    print("Finished running registration for", outFn)


def stackNiftis(origFn, registeredFns, outFn):
    """
    Combine the registered timepoint images into a single file.

    Inputs:
    - origFn: filename of the original image file
    - registeredFns: list of filenames for the registered timepoint images
    - outFn: name of the file to write the combined image to

    Returns:
    - Nothing

    Effect:
    - Combine the registered timepoint images into a single file
    """
    # load the original image
    origImg = load_image(origFn)
    # get the coordinates
    coords = origImg.coordmap
    print(origImg.get_data().shape)

    imgs = []
    # load all of the images
    for imgFn in registeredFns:
        # load the image
        img = load_image(imgFn)
        if len(img.get_data().shape) == 4:
            imgs.append(np.squeeze(img.get_data()))
        else:
            imgs.append(img.get_data())

    imgs = np.stack(imgs, axis=-1)
    print(imgs.shape)
    print(coords)
    
    registeredImg = Image(imgs, coords)
    save_image(registeredImg, outfn)
    print('Registered files merged to',outFn)

#---------------------------------------------------------------------------------
# Motion Correction: Big Functions
#---------------------------------------------------------------------------------
def motionCorrection(timepointFns, outputDir, baseDir, prealign=False):
    """
    Register each timepoint to the template image.

    Inputs:
    - timepointFns: list of filenames for each timepoint
    - outputDir: directory to write the output files to
    - prealign: default is False - do you want to prealign the nonlinear 
                registration using an affine transform?

    Outputs:
    - registeredFns: list of registered timepoint files

    Effects:
    - Writes each registered file to /path/markov-movement-correction/tmp/registered/
    """

    if not os.path.exists(outputDir+'registered/'):
        os.mkdir(outputDir+'registered/')
    # get the template image filename
    templateFn = timepointFns[0]
    # set up lists
    registeredFns = []
    myThreads = []
    # for each subsequent image
    for i in xrange(1, len(timepointFns), 1):
    # for i in xrange(1, 4, 1):
        # set the output filename
        outFn = outputDir+'registered/'+ str(i).zfill(3)+'.nii.gz'
        registeredFns.append(outFn)
        # start a thread to register the new timepoint to the template
        t = motionCorrectionThread(i, str(i).zfill(3), templateFn, timepointFns[i], outFn, outputDir, prealign=prealign)
        myThreads.append(t)
        t.start()
        # do I need to limit the number of threads?
        # or will they automatically limit themselves to the number of cores?

    for t in myThreads:
        t.join()

    return registeredFns


def markovCorrection(timepoints, outputDir, baseDir, corrId=None):
    """
    Apply the markov motion correction algorithm to a timeseries image.
    Assumes that the first filename in the timepoints list specifies the
    template image.

    Inputs:
    - timepoints: list of filenames for each timepoint
    - outputDir: directory to write the output files to
    - baseDir: base directory

    Outputs:
    - registeredFns: list of registered timepoint files

    Effects:
    - Writes each registered file to /path/markov-movement-correction/tmp/markov/
    """
    print(outputDir)
    print(baseDir)
    # # get the template image filename
    # templateFn = timepoints[0]
    # # copy the template file to the registered directory
    # shutil.copy(templateFn, outputDir)

    # # set up list
    # registeredFns = [outputDir+fn.split("/")[-1].split(".")[0]+'.nii.gz' for fn in timepoints]

    # # register the first timepoint to the template
    # registerToTemplate(templateFn, timepoints[1], registeredFns[1], outputDir, corrId=corrId)

    # # location of the transform file:
    # transformFn = outputDir+'output_InverseComposite.h5'

    # if corrId is not None:
    #     transformFn = outputDir+'output_'+str(corrId)+'_InverseComposite.h5'

    # # for each subsequent image
    # print("Number of timepoints:",len(timepoints))
    # for i in xrange(2, len(timepoints)):
    #     print("Time", i, "outfn:", registeredFns[i])
    #     # register the new timepoint to the template, using initialized transform
    #     registerToTemplate(templateFn, timepoints[i], registeredFns[i], outputDir, transformFn, corrId=corrId)

    # return registeredFns


#---------------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------------

def main(baseDir):
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Perform motion correction on time-series images.")
    # image filenames
    parser.add_argument('-i', '--inputFn', type=str, help='The name of the file to correct')
    parser.add_argument('-o', '--outputFn', type=str, help='The name of the file to save the correction to')
    # which type of motion correction
    parser.add_argument('-t', '--correctionType', type=str, help='Specify which type of correction to run. '
                        +'Options include: hmm, sequential, bi-hmm, stacking-hmm')

    # now parse the arguments
    args = parser.parse_args()
    # print(args)
    # print(args.correctionType)
    # print(args.inputFn)
    # print(args.outputFn)

    # image filename
    origFn = baseDir + args.inputFn

    # make the output directory
    baseDir = baseDir+args.inputFn.split(".")[0]+"/"
    if not os.path.exists(baseDir):
        os.mkdir(baseDir)

    # make the tmp directory
    tmpDir = baseDir+'tmp/'
    if not os.path.exists(tmpDir):
        os.mkdir(tmpDir)

    # divide the image into timepoints
    timepointFns = expandTimepoints(origFn, baseDir)

    # Select the specified motion correction algorithm
    registeredFns = []
    if args.correctionType == 'sequential':
        # make the output directory
        outputDir = baseDir+'sequential/'
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        print(outputDir)
        # register the images sequentially
        # registeredFns = motionCorrection(timepointFns, outputDir, baseDir)

    elif args.correctionType == 'hmm':
        # make the output directory
        outputDir = baseDir+'hmm/'
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        # load the template image
        img = load_image(timepointFns[0])
        coord = img.coordmap
        template = Image(img, coord)
        # save the template image in the tmp directory
        if not os.path.exists(tmpDir+"templates/"):
            os.mkdir(tmpDir+"templates/")
        save_image(template, tmpDir+"templates/hmm_"+timepointFns[0].split('/')[-1])

        print(outputDir)
        # register the images using HMM correction
        # registeredFns = markovCorrection(timepointFns, outputDir, baseDir)

    elif args.correctionType == 'bi-hmm':
        # make the output directory 
        outputDir = baseDir + "bi-hmm/"
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        print(outputDir)

        # divide the timepoint filenames list in half
        midpoint = len(timepointFns)/2
        print("midpoint:",midpoint)
        firstHalf = timepointFns[:midpoint]
        # reverse the first list
        firstHalf = firstHalf[::-1]  # reverse the first half list of filenames
        print("Check the template file:", firstHalf[0])
        secondHalf = timepointFns[midpoint-1:]
        print("Check the template file:", secondHalf[0])

        # save the template image in the tmp folder
        img = load_image(firstHalf[0])
        coord = img.coordmap
        template = Image(img, coord)
        if not os.path.exists(tmpDir+"templates/"):
            os.mkdir(tmpDir+"templates/")
        save_image(template, tmpDir+"templates/bi-hmm_"+firstHalf[0].split('/')[-1])

        # # make the threads
        # t1 = bifurcatedMarkovThread(1, "firstHalf", firstHalf, outputDir, baseDir)
        # t2 = bifurcatedMarkovThread(2, "secondHalf", secondHalf, outputDir, baseDir)

        # # start the threads
        # t1.start()
        # t2.start()

        # # join on the threads
        # out1 = t1.join()
        # #t2.start()
        # out2 = t2.join()

        # # format the filenames
        # registeredFns = list(set(out1+out2))
        # registeredFns.sort()

    elif args.correctionType == 'stacking-hmm':
        print("Currently under construction")
        # make the output directory
        outputDir = baseDir + 'stacking-hmm/'
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        # make the compartments
        numCompartments = 4
        imgsPerCompartment = np.ceil(len(timepointFns)/float(numCompartments))
        print(imgsPerCompartment)

    else:
        print("Error: the type of motion correction entered is not currently supported.")
        print("       Entered:", args.correctionType)

    # # combine the registered timepoints into 1 file
    # comboFn = baseDir+args.outputFn
    # stackNiftis(origFn, registeredFns, comboFn)


if __name__ == "__main__":
    # set the base directory
    # baseDir = '/home/pirc/Desktop/Jenna_dev/markov-movement-correction/'
    # baseDir = '/home/pirc/processing/FETAL_Axial_BOLD_Motion_Processing/markov-movement-correction/'
    #baseDir = '/home/jms565/Research/CHP-PIRC/markov-movement-correction/'
    baseDir = '/home/jenna/Research/CHP-PIRC/markov-movement-correction/'

    # very crude numpy version check
    npVer = np.__version__
    npVerList = [int(i) for i in npVer.split('.')]
    if npVerList[1] < 12:
        sys.exit("Warning: the version for numpy is "+np.__version__+".\nPlease update to at least version 1.12.1 to use this pipeline.")
    main(baseDir)

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
Perform different types of movement correction on a time series of 3D images. Current
options are
  * first-timepoint: correct all images to the first timepoint
  * hmm: follows the HMM movement correction algorithm outlined in "Temporal 
         registration in in-utero volumetric MRI time series" by Liao et al.
  * stacking-hmm: divide the time series into subcompartments, HMM each compartment,
                  and link the compartments together
  * sequential: basic, sequential alignment of each image to the previous one in the
                time series (currently needs testing)
  * template: correct all images to the image with the lowest total acquisition 
              correlation ratio
"""

#---------------------------------------------------------------------------------
# Threading Classes
#---------------------------------------------------------------------------------
class motionCorrectionThread(threading.Thread):
    """
    Implementation of the threading class.
    """
    def __init__(self, threadId, name, templateFn, timepointFn, outputFn, outputDir, templatePrefix, prealign=False, regType='nonlinear'):
        # What other properties my threads will need?
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = name
        self.templateFn = templateFn
        self.timepointFn = timepointFn
        self.outputFn = outputFn
        self.outputDir = outputDir
        self.templatePrefix = templatePrefix
        self.prealign = prealign
        self.regType = regType

    def run(self):
        print("Starting motion correction for", self.name)
        if not self.prealign:
            registerToTemplate(self.templateFn, self.timepointFn, self.outputFn, self.outputDir, self.templatePrefix, regType=self.regType)
        else:
            registerToTemplatePrealign(self.templateFn, self.timepointFn, self.outputFn, self.outputDir, self.templatePrefix, regType=self.regType)
        print("Finished motion correction for", self.name)

class hmmMotionCorrectionThread(threading.Thread):
    """
    Implementation of the threading class.

    Purpose: allow for sectioned HMM motion correction. 
    """
    def __init__(self, threadId, threadName, filenames, outputDir, transformPrefix, regType='nonlinear'):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = threadName
        self.fns = filenames
        self.outputDir = outputDir
        self.transformPrefix = transformPrefix
        self._return = None
        self.regType = regType

    def run(self):
        print("Starting the HMM motion correction for", self.name)
        outfiles = markovCorrection(self.fns, self.outputDir, self.transformPrefix, regType=self.regType)
        print("Finished the HMM motion correction for", self.name)
        self._return = outfiles

    def join(self):
        threading.Thread.join(self)
        return self._return

class linkingTransformThread(threading.Thread):
    """
    Implementation of the threading class.

    Purpose: allow for linking transforms between compartments to be performed
             in parallel 
    """
    def __init__(self, threadId, threadName, fn1, fn2, transformFn):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = threadName
        self.prevImg = fn1
        self.nextImg = fn2
        self.transformFn = transformFn

    def run(self):
        calculateLinkingTransform(self.prevImg, self.nextImg, self.transformFn)
        print("Finished thread", self.name)

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
    # print(img.get_data().shape)
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


def calculateLinkingTransform(prevCompImg, nextCompImg, transformPrefix):
    """
    Register 2 images taken at different timepoints.

    Inputs:
    - prevCompImg: filename of the last image from the previous compartment
    - nextCompImg: filename of the first image from the next compartment
    - transformPrefix: name of the file to save the transform to

    Returns:
    - None

    Effects:
    - Saves the registration files
    """

    # for debugging
    # print(prevCompImg)
    # print(nextCompImg)

    # check if the transform file exists:
    # if not os.path.isfile(transformFn+"Composite.h5") and not os.path.isfile(transformFn+"InverseComposite.h5"):
    #     print("Transform files don't exist!")

    reg = Registration()
    reg.inputs.fixed_image = prevCompImg
    reg.inputs.moving_image = nextCompImg
    reg.inputs.output_transform_prefix = transformPrefix
    reg.inputs.interpolation = 'NearestNeighbor'

    # Affine transform
    reg.inputs.transforms = ['Affine']
    reg.inputs.transform_parameters = [(2.0,)]
    reg.inputs.number_of_iterations = [[500, 20]] #1500, 200
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = False
    reg.inputs.collapse_output_transforms = True
    reg.inputs.initialize_transforms_per_stage = False
    reg.inputs.metric = ['CC']
    reg.inputs.metric_weight = [1]
    reg.inputs.radius_or_number_of_bins = [5]
    reg.inputs.sampling_strategy = ['Random']
    reg.inputs.sampling_percentage = [0.05]
    reg.inputs.convergence_threshold = [1.e-2]
    reg.inputs.convergence_window_size = [20]
    reg.inputs.smoothing_sigmas = [[2,1]]
    reg.inputs.sigma_units = ['vox']
    reg.inputs.shrink_factors = [[2,1]]

    reg.inputs.use_estimate_learning_rate_once = [True]
    reg.inputs.use_histogram_matching = [True] # This is the default
    reg.inputs.output_warped_image = False
    # reg.inputs.output_warped_image = 'testing.nii.gz'

    # print(reg.cmdline)
    print("Calculating linking transform for",transformPrefix)
    reg.run()
    print("Finished calculating linking transform for", transformPrefix)

    # else:
    #     print("WARNING: existing transform files found, linking transform calculation skipped.")


def registerToTemplate(fixedImgFn, movingImgFn, outFn, outDir, transformPrefix, initialize=False, initialRegFile=0, regType='nonlinear'):
    """
    Register 2 images taken at different timepoints.

    Inputs:
    - fixedImgFn: filename of the fixed image (should be the template image)
    - movingImgFn: filename of the moving image (should be the Jn image)
    - outFn: name of the file to write the transformed image to.
    - outDir: path to the tmp directory
    - transformPrefix: prefix for the transform function
    - initialize: optional parameter to specify the location of the
                  transformation matrix from the previous registration
    - initialRegFile: optional parameter to be used with the initialize paramter;
                      specifies which output_#Affine.mat file to use

    Outputs:
    - None

    Effects:
    - Saves the registered image and the registration files
    """
    # print("Output filename:", outFn)
    print(movingImgFn)
    # if not os.path.isfile(outFn):
    #     print("The file to be registered does not exist. Registering now.")

    # Affine and SyN transforms
    reg = Registration()
    reg.inputs.fixed_image = fixedImgFn
    # reg.inputs.moving_image = outFn
    reg.inputs.moving_image = movingImgFn
    reg.inputs.output_transform_prefix = transformPrefix
    reg.inputs.interpolation = 'NearestNeighbor'
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = False
    reg.inputs.collapse_output_transforms = False
    reg.inputs.initialize_transforms_per_stage = False

    if regType == 'nonlinear':
        reg.inputs.transforms = ['Affine', 'SyN']
        reg.inputs.transform_parameters = [(2.0,),(0.25, 3.0, 0.0)]
        reg.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]] 
        reg.inputs.metric = ['CC']*2
        reg.inputs.metric_weight = [1]*2
        reg.inputs.radius_or_number_of_bins = [5]*2
        reg.inputs.convergence_threshold = [1.e-8, 1.e-9]
        reg.inputs.convergence_window_size = [20]*2
        reg.inputs.smoothing_sigmas = [[1,0],[2,1,0]]
        reg.inputs.sigma_units = ['vox']*2
        reg.inputs.shrink_factors = [[2,1],[3,2,1]]
        reg.inputs.use_estimate_learning_rate_once = [True,True]
        reg.inputs.use_histogram_matching = [True, True] # This is the default

    elif regType == 'affine':
        reg.inputs.transforms = ['Affine']
        reg.inputs.transform_parameters = [(2.0,)]
        reg.inputs.number_of_iterations = [[1500, 200]] 
        reg.inputs.metric = ['CC'] #['MI']#['CC']   # Um, why is this mutual information and not cross correlation? I think there was a legit reason...
        reg.inputs.metric_weight = [1]
        reg.inputs.radius_or_number_of_bins = [5] # [32] #[5]
        reg.inputs.convergence_threshold = [1.e-8]
        reg.inputs.convergence_window_size = [20]
        reg.inputs.smoothing_sigmas = [[1,0]]
        reg.inputs.sigma_units = ['vox']
        reg.inputs.shrink_factors = [[2,1]]
        reg.inputs.use_estimate_learning_rate_once = [True]
        reg.inputs.use_histogram_matching = [True] # This is the default

    elif regType == 'rigid':
        reg.inputs.transforms = ['Rigid']
        reg.inputs.transform_parameters = [(2.0,)]
        reg.inputs.number_of_iterations = [[100, 20]]
        reg.inputs.metric = ['CC']  # changed to MI from CC for computational time reasons?
        reg.inputs.metric_weight = [1]
        reg.inputs.radius_or_number_of_bins = [5]
        reg.inputs.convergence_threshold = [1.e-8]
        reg.inputs.convergence_window_size = [20]
        reg.inputs.smoothing_sigmas = [[1,0]]
        reg.inputs.sigma_units = ['vox']
        reg.inputs.shrink_factors = [[2,1]]
        reg.inputs.use_estimate_learning_rate_once = [True]
        reg.inputs.use_histogram_matching = [True] # This is the default
        # get the name of the volume being registered
        volToRegister = movingImgFn.split('/')[-1]
        volNum = volToRegister.split('.')[0]
        reg.inputs.output_transform_prefix = transformPrefix+'_'+str(volNum)+'_'
        print(transformPrefix+'_'+str(volNum)+'_')

    reg.inputs.output_warped_image = outFn
    reg.inputs.num_threads = 50

    if initialize is True:
        if regType == 'rigid':
            if int(volNum) == 2:
                reg.inputs.initial_moving_transform = transformPrefix+'_'+str((int(volNum)-1)).zfill(3)+'_0Rigid.mat'
            else:
                reg.inputs.initial_moving_transform = transformPrefix+'_'+str((int(volNum)-1)).zfill(3)+'_1Rigid.mat'
        else:
            reg.inputs.initial_moving_transform = transformPrefix+str(initialRegFile)+'Affine.mat'
        reg.inputs.invert_initial_moving_transform = False

    # print(reg.cmdline)
    print("Starting", regType, "registration for",outFn)
    reg.run()
    print("Finished", regType, "registration for",outFn)

    # tmpIdx = transformPrefix.index('transforms/')+len('transforms/')
    # transformDir = os.listdir(transformPrefix[:tmpIdx])
    # transformDir.sort()
    # print("Files in transform/ dir:")
    # for fn in transformDir:
    #     print("   ", fn)

    # else:
    #     print("WARNING: existing registered image found, image registration skipped.")


def alignCompartments(fixedImg, movingImgs, transform):
    """
    Given a precalculated linking transform and a fixed image (used for coordinate
    system and/or metadata), align each image in the movingImgs list to the fixed image.

    Inputs:
    - fixedImg: the path to the fixed image
    - movingImgs: a list of paths to the moving images
    - transform: either a list of paths or a single path to a transform file

    Returns:
    - None

    Effects:
    - Overwrites the specified images with a more aligned version of the same images

    """
    # for each image
    for m in movingImgs:
        # set up the transform application
        at = ApplyTransforms()
        at.inputs.input_image = m
        at.inputs.reference_image = fixedImg
        at.inputs.output_image = m
        at.inputs.transforms = transform+'_1Affine.mat'
        at.inputs.interpolation = 'NearestNeighbor'
        at.inputs.invert_transform_flags =[False]
        # run the transform application
        at.run()


def stackNiftis(origFn, registeredFns, outFn):
    """
    Combine the list of registered timepoint images into a single file.

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
    # print(imgs.shape)
    # print(coords)
    
    registeredImg = Image(imgs, coords)
    save_image(registeredImg, outFn)
    print('Registered files merged to',outFn)

#---------------------------------------------------------------------------------
# Motion Correction: Big Functions
#---------------------------------------------------------------------------------
def motionCorrection(templateFn, timepointFns, outputDir, baseDir, prealign=False, regType='nonlinear'):
    """
    Register each timepoint to the template image.

    Inputs:
    - templateFn: the filename of the template image
    - timepointFns: list of filenames for each timepoint
    - outputDir: directory to write the output files to
    - prealign: default is False - do you want to prealign the nonlinear 
                registration using an affine transform?

    Outputs:
    - registeredFns: list of registered timepoint files

    Effects:
    - Writes each registered file to /path/markov-movement-correction/tmp/registered/
    """

    # set up lists
    registeredFns = []
    myThreads = []
    # for each subsequent image
    for i in xrange(len(timepointFns)):
        if timepointFns[i] == templateFn:
            # copy the template file into the output directory
            shutil.copy2(templateFn, outputDir)
        else:
            # set the output filename
            outFn = outputDir+str(i).zfill(3)+'.nii.gz'
            registeredFns.append(outFn)
            templatePrefix = baseDir+'tmp/output_'
            # start a thread to register the new timepoint to the template
            t = motionCorrectionThread(i, str(i).zfill(3), templateFn, timepointFns[i], outFn, outputDir, templatePrefix, prealign=prealign, regType=regType)
            myThreads.append(t)
            t.start()
        # do I need to limit the number of threads?

    # print(timepointFns)

    for t in myThreads:
        t.join()

    return registeredFns


def markovCorrection(timepoints, outputDir, transformPrefix, regType='nonlinear'):
    """
    Apply the markov motion correction algorithm to a timeseries image.
    Assumes that the first filename in the timepoints list specifies the
    template image.

    Inputs:
    - timepoints: list of filenames for each timepoint
    - outputDir: directory to write the output files to
    - transformPrefix: prefix for the transform files

    Outputs:
    - registeredFns: list of registered timepoint files

    Effects:
    - Writes each registered file to /path/markov-movement-correction/tmp/markov/
    """
    # print(outputDir)
    # get the template image filename
    templateFn = timepoints[0]
    # copy the template file to the registered directory
    shutil.copy(templateFn, outputDir)

    # set up list
    registeredFns = [outputDir+fn.split("/")[-1].split(".")[0]+'.nii.gz' for fn in timepoints]

    # location of the transform file:
    # print("In markovCorrection (prefix):", transformPrefix)

    # register the first timepoint to the template
    registerToTemplate(templateFn, timepoints[1], registeredFns[1], outputDir, transformPrefix, initialize=False, regType=regType)

    # register the second timepoint to the template using the initialized transform
    registerToTemplate(templateFn, timepoints[2], registeredFns[2], outputDir, transformPrefix, initialize=True, initialRegFile=0, regType=regType)

    # for each subsequent image
    # print("Number of timepoints:",len(timepoints))
    for i in xrange(3, len(timepoints)):
        # print("Time", i, "outfn:", registeredFns[i])
        # register the new timepoint to the template, using initialized transform
        registerToTemplate(templateFn, timepoints[i], registeredFns[i], outputDir, transformPrefix, initialize=True, initialRegFile=1, regType=regType)

    return registeredFns


def stackingHmmCorrection(origTimepoints, baseDir, numCompartments, regType='nonlinear'):
    """
    Perform stacking-hmm correction on the filenames passed to the function.

    Inputs:
    - origTimepoints: the names of the original files
    - baseDir: the base directory
    - numCompartments: the number of compartments to use

    Returns:
    - registeredFns: a list of registered filenames

    Effects:
    - 
    """

    # print("There are", numCompartments,"compartments being used.")

    # make the output directory
    outputDir = baseDir + 'stacking-hmm/'
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    # tmpDir is not a global variable
    tmpDir = baseDir + 'tmp/'
    if not os.path.exists(tmpDir):
        os.mkdir(tmpDir)

    # Step 1: Divide the time series into compartments
    timepointFns = origTimepoints #list(reversed(origTimepoints))
    imgsPerCompartment = int(np.ceil(len(timepointFns)/float(numCompartments)))
    # make the list of lists
    compartments = [timepointFns[i*imgsPerCompartment:((i+1)*imgsPerCompartment)+1] for i in xrange(numCompartments-1)]
    compartments.append(timepointFns[imgsPerCompartment*(numCompartments-1):])
    # print("There are", len(compartments))

    flatCompartments = compartments[:]
    for i in xrange(len(compartments)):
        if i == 0:
            flatCompartments[i] = flatCompartments[i]
        else:
            flatCompartments[i] = flatCompartments[i][1:]

    # for c in compartments:
    #     print(c)
    #     print(len(c))

    # print()

    # for c in flatCompartments:
    #     print(c)
    #     print(len(c))

    # Step 2: perform regular HMM motion correction in each compartment
    # set up the variable to indicate the location of the transform prefix
    threads = []
    if not os.path.exists(tmpDir+"compartmentTransforms/"):
        os.mkdir(tmpDir+"compartmentTransforms/")
    transformPrefix = tmpDir+"compartmentTransforms/stacking-hmm_"
    compartmentTransformFns = []
    # iterate over all compartments
    for i in xrange(len(compartments)):
        # make a new HMM motion correction thread
        t = hmmMotionCorrectionThread(i, "compartment_"+str(i), compartments[i], outputDir, transformPrefix+str(i)+'_', regType=regType)
        # add the name of the transform file to the appropriate list
        compartmentTransformFns.append(transformPrefix+str(i))
        # print("In stackingHmmCorrection:", transformPrefix+str(i))
        # add the thread to the list of threads
        threads.append(t)

    # start the threads
    for t in threads:
        t.start()

    # join on the threads
    hmmCompartments = []
    for t in threads:
        hmmCompartments.append(t.join())

    # print("Number of compartment transform filenames:",len(compartmentTransformFns))
    # sort the hmmCompartments
    hmmCompartments = sorted(hmmCompartments)

    # Step 4: apply linking transform to each compartment
    # iterate through the transform functions list, backward
    # skip the last transform function - it's for the last compartment
    # and doesn't help align the compartments
    for i in xrange(len(compartmentTransformFns)-2, -1, -1):
        # get a flat list of images to correct, starting with the last compartment
        imgsToAlign = [img for compartment in flatCompartments[i+1:] for img in compartment]
        alignCompartments(origTimepoints[0], imgsToAlign, compartmentTransformFns[i])


    registeredFns = [img for compartment in flatCompartments for img in compartment]
    # # print(compartmentTransformFns)
    # # print(linkingTransFns)
    # print("Number of aligned files:", len(alignedFns))
    return registeredFns

#---------------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------------

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Perform motion correction on time-series images.")
    # image filenames
    parser.add_argument('-i', '--inputFn', type=str, help='Full path to the name of the file to correct')
    parser.add_argument('-o', '--outputFn', type=str, help='The name of the file to save the correction to (within the directory containing the original file.')
    # which type of global volume registration framework
    parser.add_argument('-t', '--correctionType', type=str, help='Specify which type of correction to run. '
                        +'Options include: first-timepoint, template, sequential, hmm, stacking-hmm, and testing (use at your own risk)')
    # which type of registration
    parser.add_argument('-r', '--registrationType', type=str, help='Specify which type of registration to use: rigid, affine, or nonlinear')

    # now parse the arguments
    args = parser.parse_args()

    # image filename
    origFn = args.inputFn.replace("//", '/')
    baseDir = origFn.rsplit("/", 1)[0]+'/'
    print(origFn)
    print(baseDir)

    # # make the output directory
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

    if args.correctionType == 'first-timepoint':
        """
        First timepoint matching: align each image to the first timepoint
        """
        # make the output directory
        outputDir = baseDir+'firstTimepointMatching/'
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        # register the images sequentially
        templateImg = timepointFns[0]
        registeredFns = motionCorrection(templateImg, timepointFns, outputDir, baseDir, regType=args.registrationType)

    elif args.correctionType == 'template':
        """
        Template matching: find the image that is the most similar to all other images,
                           then align each image to it
        """
        # make the output directory
        outputDir = baseDir + 'templateMatching/'
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        # find the template image
        # calculate the total similarity of every image to every other image (excluding self)
        similarities = [0.0] * len(timepointFns)
        idx = 0
        for template in timepointFns:
            imgTotalSims = 0.0
            for img in timepointFns:
                if not img == template:
                    sim = Similarity()
                    sim.inputs.volume1 = template
                    sim.inputs.volume2 = img
                    sim.inputs.metric = 'cr'
                    res = sim.run()
                    imgTotalSims += res.outputs.similarity[0]
            similarities[idx] = imgTotalSims
            idx += 1
        # find the minimum total similarity
        minSim, minLoc = min((val, idx) for (idx, val) in enumerate(similarities))
        maxSim, maxLoc = max((val, idx) for (idx, val) in enumerate(similarities))
        print("Min similarity", minSim, "at", minLoc)
        print("Max similarity", maxSim, "at", maxLoc)
        # save the location of the template to a file
        fn = baseDir+'templateMatching-templateName.txt'
        with open(fn, 'w') as f:
            f.write(str(minLoc))

        # define the template image
        templateImg = timepointFns[minLoc]

        # run the motion correction
        registeredFns = motionCorrection(templateImg, timepointFns, outputDir, baseDir, regType=args.registrationType)

    elif args.correctionType == 'sequential':
        """
        Sequential: align each image to the previous image
        """
        # make the output directory
        outputDir = baseDir+'sequential/'
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        # make the transform prefix
        transformPrefix = baseDir+'tmp/sequential_'
        if not os.path.exists(baseDir+'tmp/'):
            os.mkdir(baseDir+'tmp/')

        # # copy the first image to the output directory
        shutil.copy(timepointFns[0], outputDir)
        # print(timepointFns[0])
        registeredFns.append(outputDir+'000.nii.gz')

        # register the second image to the first
        outFn = outputDir+'001.nii.gz'
        registerToTemplate(timepointFns[0], timepointFns[1], outFn, outputDir, transformPrefix, initialize=False, regType=args.registrationType)
        registeredFns.append(outFn)

        # for every image
        # for i in xrange(2, 3, 1):
        for i in xrange(2, len(timepointFns), 1):
            # set the output filename
            outFn = outputDir+str(i).zfill(3)+'.nii.gz'
            # register the image to the previous image
            # print(registeredFns[-1])
            registerToTemplate(registeredFns[-1], timepointFns[i], outFn, outputDir, transformPrefix, initialize=False, regType=args.registrationType)
            registeredFns.append(outFn)

        # print(registeredFns)

    elif args.correctionType == 'hmm':
        """
        HMM: as proposed in MIT's paper
        """
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

        # set up the variable to indicate the location of the transform prefix
        if not os.path.exists(tmpDir+"prealignTransforms/"):
            os.mkdir(tmpDir+"prealignTransforms/")
        transformPrefix = tmpDir+"prealignTransforms/hmm_"

        # print(outputDir)
        # print(transformPrefix)
        # register the images using HMM correction
        registeredFns = markovCorrection(timepointFns, outputDir, transformPrefix, regType=args.registrationType)

    elif args.correctionType == 'stacking-hmm':
        """
        Stacking HMM: divide the timeseries into compartments, HMM each compartment, recombine
        """
        # make compartments
        numCompartments = 5 # 150/6 = 25, nice and even
        registeredFns = stackingHmmCorrection(timepointFns, baseDir, numCompartments)
        
    elif args.correctionType == 'testing':
        """
        Testing #1: get stacking-hmm working and producing good results
                    currently adding lots of extra files/dirs for testing
        """
        # get a subset of images
        subset = timepointFns
        print(baseDir)
        # # make a testing dir
        testDir = baseDir+'testing/'
        if not os.path.exists(testDir):
            os.mkdir(testDir)

        # ## First timepoint template
        # outDir = testDir+'first/'
        # if not os.path.exists(outDir):
        #     os.mkdir(outDir)

        # templateImg = subset[0]
        # registeredFns = motionCorrection(templateImg, subset, outDir, baseDir)

        # ## HMM
        # outDir = testDir+'hmm/'
        # if not os.path.exists(outDir):
        #     os.mkdir(outDir)

        # if not os.path.exists(testDir+'transforms/'):
        #     os.mkdir(testDir+'transforms/')

        # registeredFns = markovCorrection(subset, outDir, testDir+'transforms/testing_hmm_transform_')

        # Stacking Markov
        # copy the subset to a timepoints dir in testing dir
        spareDir = testDir+"timepoints/"
        if not os.path.exists(spareDir):
            os.mkdir(spareDir)
        for img in subset:
            shutil.copy2(img, spareDir)
        subset = [img.replace('timepoints/', 'testing/timepoints/') for img in subset]
        
        # now use the stacking-hmm function
        numCompartments = 6
        print("Submitting", numCompartments, "compartments")
        registeredFns = stackingHmmCorrection(subset, testDir, numCompartments, regType=args.registrationType)

    else:
        print("Error: the type of motion correction entered is not currently supported.")
        print("       Entered:", args.correctionType)

    # combine the registered timepoints into 1 file
    comboFn = baseDir+args.outputFn
    stackNiftis(origFn, registeredFns, comboFn)

    return origFn, args.correctionType


if __name__ == "__main__":
    # very crude numpy version check
    npVer = np.__version__
    npVerList = [int(i) for i in npVer.split('.')]
    if npVerList[1] < 12:
        sys.exit("Warning: the version for numpy is "+np.__version__+".\nPlease update to at least version 1.12.1 to use this pipeline.")
        
    startTime = time.time()
    subjFn, method = main()
    endTime = time.time() - startTime

    # turn this into a function
    totalDays = np.floor(endTime/24.0/60.0/60.0)
    endTime = endTime%(24.0*60.0*60.0)
    totalHours = np.floor(endTime/60.0/60.0)
    endTime = endTime%(60.0*60.0)
    totalMins = np.floor(endTime/60.0)
    totalSecs = endTime%60.0
    print("Total run time:",totalDays,"days,",totalHours,"hours,",totalMins,"minutes,",totalSecs)

    # set up the lines and variables to write for the runtime analysis
    headerLine = "Subject, Method, RunTime (DD:HH:MM:SS)\n"
    subj = subjFn.split("/")[-2]
    baseDir = subjFn.split(subj)[0]
    print(baseDir)
    timeLine = subj+", "+ method+", "+ str(totalDays).zfill(2) + ":" + str(totalHours).zfill(2) + ":" + str(totalMins).zfill(2) + ":" + str(totalSecs).zfill(2) + "\n"

    # write the time to a file
    fn = baseDir+"timeToRun.csv"
    if not os.path.isfile(fn):
        # open the file in write
        with open(fn, "w") as file:
            # write the header line
            file.write(headerLine)
            # write the time line
            file.write(timeLine)

    else:
        with open(fn, "a+") as file:
            # write the time line
            file.write(timeLine)

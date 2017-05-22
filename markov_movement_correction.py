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

"""
Currently a test file. Eventual purpose is to perform movement correction on time series images.
"""

#---------------------------------------------------------------------------------
# Threading Class
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

    # pull out the first image, timepoint 0, as the template
    template = img[:,:,:,0].get_data()[:,:,:,None]
    template_img = Image(template, coord)
    save_image(template_img, outDir+'template.nii.gz')
    # also save the first image as 000, but don't add the name to the list
    save_image(template_img, outDir+str(0).zfill(3)+'.nii.gz')

    # build the list of filenames
    filenames = [outDir+'template.nii.gz']

    # for the remaining images
    for i in xrange(1, img.get_data().shape[3], 1):
        # pull out the image and save it
        tmp = img[:,:,:,i].get_data()[:,:,:,None]
        tmp_img = Image(tmp, coord)
        outFn = str(i).zfill(3)+'.nii.gz'
        save_image(tmp_img, outDir+outFn)
        filenames.append(outDir+outFn)

    return filenames


def registerToTemplate(fixedImgFn, movingImgFn, outFn, outDir, initialize=None):
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
        reg.inputs.invert_initial_moving_transform = True

    # print(reg.cmdline)
    print("Starting registration for",outFn)
    reg.run()
    print("Finished running registration for", outFn)


def registerToTemplatePrealign(fixedImgFn, movingImgFn, outFn, outDir, initialize=None):
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
    reg = Registration()
    reg.inputs.fixed_image = fixedImgFn
    reg.inputs.moving_image = movingImgFn
    reg.inputs.output_transform_prefix = outDir+"output_"
    reg.inputs.transforms = ['Affine', 'SyN']
    reg.inputs.transform_parameters = [(2.0, 0.5, 0.0), (0.25, 3.0, 0.0)]
    reg.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]]
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = False
    reg.inputs.initialize_transforms_per_stage = False
    reg.inputs.metric = ['CC']*2
    reg.inputs.metric_weight = [1]*2 # Default (value ignored currently by ANTs)
    reg.inputs.radius_or_number_of_bins = [32]*2
    reg.inputs.sampling_strategy = ['Random', None]
    reg.inputs.sampling_percentage = [0.05, None]
    reg.inputs.convergence_threshold = [1.e-8, 1.e-9]
    reg.inputs.convergence_window_size = [20]*2
    reg.inputs.smoothing_sigmas = [[1,0], [2,1,0]]  # probably should fine-tune these?
    reg.inputs.sigma_units = ['vox'] * 2
    reg.inputs.shrink_factors = [[2,1], [3,2,1]]  # probably should fine-tune these?
    reg.inputs.use_estimate_learning_rate_once = [True, True]
    reg.inputs.use_histogram_matching = [True, True] # This is the default
    reg.inputs.output_warped_image = outFn

    if initialize is not None:
        reg.inputs.initial_moving_transform = initialize
        reg.inputs.invert_initial_moving_transform = True

    # print(reg.cmdline)
    print("Starting registration for",outFn)
    reg.run()
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
        outFn = outputDir'registered/'+ str(i).zfill(3)+'.nii.gz'
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


def markovCorrection(timepoints, outputDir, baseDir):
    """
    Apply the markov motion correction algorithm to a timeseries image.

    Inputs:
    - timepointFns: list of filenames for each timepoint
    - outputDir: directory to write the output files to
    - baseDir: base directory

    Outputs:
    - registeredFns: list of registered timepoint files

    Effects:
    - Writes each registered file to /path/markov-movement-correction/tmp/markov/
    """
    if not os.path.exists(outputDir+'markov/'):
        os.mkdir(outputDir+'markov/')
    # get the template image filename
    templateFn = timepoints[0]
    # set up list: want the original 000.nii.gz file
    registeredFns = [outputDir+'timepoints/'+str(0).zfill(3)+'.nii.gz']

    # register the first timepoint to the template
    outFn = outputDir+'markov/'+ str(1).zfill(3)+'.nii.gz'
    registerToTemplate(templateFn, timepoints[1], outFn, outputDir)

    # location of the transform file:
    transformFn = outputDir+'output_InverseComposite.h5'

    # for each subsequent image
    for i in xrange(2, len(timepoints), 1):
    # for i in xrange(2, 3, 1):
        # set the output filename
        outFn = outputDir+'markov/'+ str(1).zfill(3)+'.nii.gz'
        registeredFns.append(outFn)
        # register the new timepoint to the template, using initialized transform
        registerToTemplate(templateFn, timepoints[i], outFn, outputDir, transformFn)

    return registeredFns

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
                        +'Options include: markov, non-markov, non-markov-affine')

    # now parse the arguments
    args = parser.parse_args()
    print(args)
    print(args.correctionType)
    print(args.inputFn)
    print(args.outputFn)

    # image filename
    #imgFn = baseDir + '0003_MR1/scans/4/18991230_000000EP2DBOLDLINCONNECTIVITYs004a001.nii.gz'
    # imgFn = baseDir + '0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001.nii.gz'
    # imgFn = baseDir + '0003_MR2/scans/4/18991230_000000EP2DBOLDLINCONNECTIVITYs004a001.nii.gz'
    origFn = baseDir + args.inputFn

    # make the tmp directory
    outputDir = baseDir+"tmp/"+args.inputFn.split(".")[0]+"/"
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    # divide the image into timepoints
    timepointFns = expandTimepoints(origFn, outputDir)

    # Select the specified motion correction algorithm
    registeredFns = []
    if args.correctionType == 'non-markov':
        registeredFns = motionCorrection(timepointFns, outputDir, baseDir)
    elif args.correctionType == 'non-markov-affine':
        registeredFns = motionCorrection(timepointFns, outputDir, baseDir, prealign=True)
    elif args.correctionType == 'markov':
        registeredFns = markovCorrection(timepointFns, outputDir, baseDir)
    else:
        print("Error: the type of motion correction entered is not currently supported.")
        print("       Entered:", args.correctionType)

    """
    # Motion correction to template: register all timepoints to the template image (timepoint 0)
    # registeredFns = motionCorrection(timepointFns, outputDir, baseDir)
    # comboFn = baseDir+'tmp/motion_registered_0003_MR1'

    # Markov motion correction: register all timepoints to preregistered
    # registeredFns = markovCorrection(timepointFns, outputDir, baseDir)
    # comboFn = baseDir+'tmp/markov_registered_0003_MR1'
    """

    # combine the registered timepoints into 1 file
    comboFn = baseDir+args.outputFn
    stackNiftis(origFn, registeredFns, comboFn)

#------------------------------------------------------------------------------------------
    """
    # load the image
    img = load_image(imgFn)
    coord = img.coordmap
    # get the images for the 3 time points of current interest
    template = img[:,:,:,0].get_data()[:,:,:,None]
    slightMovement = img[:,:,:,1].get_data()[:,:,:,None]
    lotsMovement = img[:,:,:,144].get_data()[:,:,:,None]
    """
    # when finished, remove the tmp directory
#-----------------------------------------------------------------------------------------


if __name__ == "__main__":
    # set the base directory
    # baseDir = '/home/pirc/Desktop/Jenna_dev/markov-movement-correction/'
    baseDir = '/home/pirc/processing/FETAL_Axial_BOLD_Motion_Processing/markov-movement-correction/'
    #baseDir = '/home/jms565/Research/CHP-PIRC/markov-movement-correction/'
    #baseDir = '/home/jenna/Research/CHP-PIRC/markov-movement-correction/'
    main(baseDir)
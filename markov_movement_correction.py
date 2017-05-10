from __future__ import print_function
from nipy import load_image, save_image
import numpy as np
import getpass
from nipy.core.api import Image
import os

# for the registration
from nipype.interfaces.ants import Registration

# for saving the registered file
from nipype.interfaces import dcmstack

# threading
import threading
import time

"""
Currently a test file. Eventual purpose is to perform movement correction on time series images.
"""

#---------------------------------------------------------------------------------
# Threading Classes
#---------------------------------------------------------------------------------
class motionCorrectionThread(threading.Thread):
    """
    Implementation of the threading class.
    """
    def __init__(self, threadId, name, templateFn, timepointFn, outputFn, outputDir):
        # What other properties my threads will need?
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = name
        self.templateFn = templateFn
        self.timepointFn = timepointFn
        self.outputFn = outputFn
        self.outputDir = outputDir

    def run(self):
        print("Starting motion correction for", self.name)
        registerToTemplate(self.templateFn, self.timepointFn, self.outputFn, self.outputDir)
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


def registerToTemplate(fixedImgFn, movingImgFn, outputFn, outputDir):
    """
    Register 2 images taken at different timepoints.

    Inputs:
    - fixedImgFn: filename of the fixed image (should be the template image)
    - movingImgFn: filename of the moving image (should be the Jn image)
    - outputFn: name of the file to write the transformed image to.
    - outputDir: path to the tmp directory

    Outputs:
    - Currently, nothing. Should return or save the transformation

    Effects:
    - Saves the registered image
    """
    reg = Registration()
    reg.inputs.fixed_image = fixedImgFn
    reg.inputs.moving_image = movingImgFn
    reg.inputs.output_transform_prefix = outputDir+"output_"
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
    reg.inputs.output_warped_image = outputFn
    print(reg.cmdline)
    reg.run()
    print("Finished running registration!")


def applyTransformToTemplate(templateImg, transform):
    """
    Use ANTs to apply the inverse of the transform to the template image.

    Previously, the Jn image (movement image) was transformed to match the
    template image. Now we want to transform the template image to match Jn
    so that the template will be close to Jn+1.

    Inputs:
    - templateImg:
    - transform: the transformation from Jn to template

    """
    pass


#---------------------------------------------------------------------------------
# Motion Correction: Big Functions
#---------------------------------------------------------------------------------
def motionCorrection(timepointFns, outputDir):
    """
    Register each timepoint to the template image.

    Inputs:
    - timepointFns: list of filenames for each timepoint
    - outputDir: directory to write the output files to

    Outputs:
    - registeredFns: list of registered timepoint files

    Effects:
    - Writes each registered file to /path/markov-movement-correction/tmp/registered/
    """

    if not os.path.exists(baseDir+'tmp/registered/'):
        os.mkdir(baseDir+'tmp/registered/')
    # get the template image filename
    templateFn = timepointFns[0]
    # set up lists
    registeredFns = []
    myThreads = []
    # for each subsequent image
    for i in xrange(1, len(timepointFns), 1):
    # for i in xrange(1, 4, 1):
        # set the output filename
        outFn = baseDir+'tmp/registered/'+ str(i).zfill(3)+'.nii.gz'
        registeredFns.append(outFn)
        outputDir = baseDir + 'tmp/'
        # start a thread to register the new timepoint to the template
        t = motionCorrectionThread(i, str(i).zfill(3), templateFn, timepointFns[i], 
                     outFn, outputDir)
        myThreads.append(t)
        t.start()
        # do I need to limit the number of threads?
        # or will they automatically limit themselves to the number of cores?

    for t in myThreads:
        t.join()
        
    return registeredFns


# Is this line necessary? What does it do?
user=getpass.getuser()

# set the base directory
#baseDir = '/home/pirc/Desktop/Jenna_dev/'
baseDir = '/home/jms565/Research/CHP-PIRC/markov-movement-correction/'
#baseDir = '/home/jenna/Research/CHP-PIRC/markov-movement-correction/'

# image filename
#imgFn = baseDir + '0003_MR1/scans/4/18991230_000000EP2DBOLDLINCONNECTIVITYs004a001.nii.gz'
imgFn = baseDir + '0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001.nii.gz'
# imgFn = baseDir + '0003_MR2/scans/4/18991230_000000EP2DBOLDLINCONNECTIVITYs004a001.nii.gz'

outputDir = baseDir+"tmp/"
# make the tmp directory
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

# divide the image into timepoints
timepointFns = expandTimepoints(imgFn, baseDir+'tmp/')

# register all timepoints to the template image (timepoint 0)
registeredFns = motionCorrection(timepointFns, outputDir)

# combine the registered timepoints into 1 file
niftiMerger = dcmstack.MergeNifti()
niftiMerger.inputs.in_files = registeredFns
niftiMerger.inputs.out_path = baseDir+'tmp/registered_0003_MR1'
niftiMerger.run()

#------------------------------------------------------------------------------------------
"""
# load the image
img = load_image(imgFn)
coord = img.coordmap
# get the images for the 3 time points of current interest
template = img[:,:,:,0].get_data()[:,:,:,None]
slightMovement = img[:,:,:,1].get_data()[:,:,:,None]
lotsMovement = img[:,:,:,144].get_data()[:,:,:,None]
print template.shape
template_img = Image(template, coord)
save_image(template_img, baseDir+'tmp/template.nii.gz')
slight_img = Image(slightMovement, coord)
save_image(template_img, baseDir+'tmp/slight.nii.gz')
lots_img = Image(lotsMovement, coord)
save_image(lots_img, baseDir+'tmp/lots.nii.gz')

# register the timepoints
outputFn = baseDir+'tmp/slight_transformed.nii.gz'
registerToTemplate(baseDir+'tmp/template.nii.gz', baseDir+'tmp/slight.nii.gz', outputFn, baseDir+'tmp/')

# save the template and the transformed image to the same .nii.gz file
comboImgFn = baseDir+'tmp/template_and_transformed'
comboFiles = [baseDir+'tmp/template.nii.gz', baseDir+'tmp/slight_transformed.nii.gz']
niftiMerger = dcmstack.MergeNifti()
niftiMerger.inputs.in_files = comboFiles
niftiMerger.inputs.out_path = comboImgFn
niftiMerger.run()
"""

# when finished, remove the tmp directory

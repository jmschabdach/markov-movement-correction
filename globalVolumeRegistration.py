"""
Jenna Schabdach 2018

Perform global volume registration on the specified series of 3D images
images. Currently supports the following volume registration methods:
    - traditional: register all image volumes to the first volume in 
                   the sequence
    - dag: register all image volumes to the first volume using the
           DAG-based prealignment paradigm

Can perform either affine or nonlinear registration

Useage:
    python globalVolumeRegistration.py -i [full path to input image]
    -o [filename for registered image] -t [traditional or dat, type
    of global volume registration] -r [affine or nonlinear,
    registration type]

"""

from __future__ import print_function
import numpy as np
import os
import argparse
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

#-------------------------------------------------------------------------------
# Threading Classes
#-------------------------------------------------------------------------------
class traditionalRegistrationThread(threading.Thread):
    """
    Implementation of the threading class.

    Purpose: parallelize the traditional registration process
    """
    def __init__(self, threadId, name, templateFn, timepointFn, outputFn, outputDir, templatePrefix, prealign=False, regType='nonlinear'):
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
        print("Starting traditional registration for", self.name)
        if not self.prealign:
            registerToTemplate(self.templateFn, self.timepointFn, self.outputFn, self.outputDir, self.templatePrefix, regType=self.regType)
        else:
            registerToTemplatePrealign(self.templateFn, self.timepointFn, self.outputFn, self.outputDir, self.templatePrefix, regType=self.regType)
        print("Finished traditional registration for", self.name)

class dagRegistrationThread(threading.Thread):
    """
    Implementation of the threading class.

    Purpose: allow for sectioned DAG-based volume registration.
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
        print("Starting the DAG-based registration for", self.name)
        outfiles = dagCorrection(self.fns, self.outputDir, self.transformPrefix, regType=self.regType)
        print("Finished the DAG-based registration for", self.name)
        self._return = outfiles

    def join(self):
        threading.Thread.join(self)
        return self._return

#-------------------------------------------------------------------------------
# Motion Correction: Helper Functions
#-------------------------------------------------------------------------------
def expandTimepoints(imgFn, baseDir):
    """
    Expand an image sequence stored as a .nii.gz file into a collection of 
    .nii.gz images (where each frame is its own .nii.gz file)

    Inputs:
    - imgFn: the time series image's filename
    - baseDir: the directory in which a new directory 
        will be created to hold the collection of files

    Returns:
    - filenames: list of filenames
    """
    # load the image
    img = load_image(imgFn)
    coord = img.coordmap

    if not os.path.exists(baseDir+'timepoints/'):
        os.mkdir(baseDir+'timepoints/')
    outDir = baseDir +'timepoints/'

    # pull out the first image from the sequence (timepoint 0)
    first = img[:,:,:,0].get_data()[:,:,:,None]
    first_img = Image(first, coord)
    # save the first image as 000
    save_image(first_img, outDir+str(0).zfill(3)+'.nii.gz')
    # build the list of filenames
    filenames = [outDir+'000.nii.gz']

    # for the remaining images
    for i in range(1, img.get_data().shape[3], 1):
        # pull out the image and save it
        tmp = img[:,:,:,i].get_data()[:,:,:,None]
        tmp_img = Image(tmp, coord)
        outFn = str(i).zfill(3)+'.nii.gz'
        save_image(tmp_img, outDir+outFn)
        # add the name of the image to the list of filenames
        filenames.append(outDir+outFn)

    return filenames


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
    - regType: optional parameter to specify the type of registration to use
        (affine ['Affine'] or nonlinear ['SyN'])

    Outputs:
    - None

    Effects:
    - Saves the registered image and the registration files
    """
    # Set up the registration
    # For both Affine and SyN transforms
    reg = Registration()
    reg.inputs.fixed_image = fixedImgFn
    reg.inputs.moving_image = movingImgFn
    reg.inputs.output_transform_prefix = transformPrefix
    reg.inputs.interpolation = 'NearestNeighbor'
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = False
    reg.inputs.collapse_output_transforms = False
    reg.inputs.initialize_transforms_per_stage = False
    reg.inputs.num_threads = 100

    # Specify certain parameters for the nonlinear/['SyN'] registration
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
        reg.inputs.use_histogram_matching = [True, True] # This is the default value, but specify it anyway

    # Specify certain parameters for the affine/['Affine'] registration
    elif regType == 'affine':
        reg.inputs.transforms = ['Affine']
        reg.inputs.transform_parameters = [(2.0,)]
        reg.inputs.number_of_iterations = [[1500, 200]] 
        reg.inputs.metric = ['CC'] 
        reg.inputs.metric_weight = [1]
        reg.inputs.radius_or_number_of_bins = [5] 
        reg.inputs.convergence_threshold = [1.e-8]
        reg.inputs.convergence_window_size = [20]
        reg.inputs.smoothing_sigmas = [[1,0]]
        reg.inputs.sigma_units = ['vox']
        reg.inputs.shrink_factors = [[2,1]]
        reg.inputs.use_estimate_learning_rate_once = [True]
        reg.inputs.use_histogram_matching = [True] # This is the default, but specify it anyway

    reg.inputs.output_warped_image = outFn
    reg.inputs.num_threads = 50

    # If the registration is initialized, set a few more parameters
    if initialize is True:
        reg.inputs.initial_moving_transform = transformPrefix+str(initialRegFile)+'Affine.mat'
        reg.inputs.invert_initial_moving_transform = False

    # Keep the user updated with the status of the registration
    print("Starting", regType, "registration for",outFn)
    # Run the registration
    reg.run()
    # Keep the user updated with the status of the registration
    print("Finished", regType, "registration for",outFn)


def stackNiftis(origFn, registeredFns, outFn):
    """
    Combine the list of registered image frames into a single image sequence file.

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
    # get the coordinates of that image
    coords = origImg.coordmap
    print(origImg.get_data().shape)

    imgs = []
    # load all of the images
    for imgFn in sorted(registeredFns):
        # load a single image
        img = load_image(imgFn)
        # get the contents of the image
        if len(img.get_data().shape) == 4:
            imgs.append(np.squeeze(img.get_data()))
        else:
            imgs.append(img.get_data())

    # stack the image data
    imgs = np.stack(imgs, axis=-1)
   
    # save the image sequence
    registeredImg = Image(imgs, coords)
    save_image(registeredImg, outFn)
    print('Registered files merged to',outFn)

#-------------------------------------------------------------------------------
# Volume Registration: Big Functions
#-------------------------------------------------------------------------------
def volumeRegistration(templateFn, timepointFns, outputDir, baseDir, prealign=False, regType='nonlinear'):
    """
    Register each image frame to the template image.

    Inputs:
    - templateFn: the filename of the template image
    - timepointFns: list of filenames for each timepoint
    - outputDir: directory to write the output files to
    - prealign: default is False - do you want to prealign the nonlinear 
                registration using an affine transform?

    Outputs:
    - registeredFns: list of registered timepoint files

    Effects:
    - Writes each registered file to /path/dag-movement-correction/tmp/registered/
    """

    # set up lists
    registeredFns = []
    myThreads = []
    # for each subsequent image
    for i in range(len(timepointFns)):
        if timepointFns[i] == templateFn:
            # copy the template file into the output directory
            shutil.copy2(templateFn, outputDir)
            print("FOUND THE TEMPLATE FILE")
        else:
            # set the output filename
            outFn = outputDir+str(i).zfill(3)+'.nii.gz'
            registeredFns.append(outFn)
            templatePrefix = baseDir+'tmp/output_'
            # start a thread to register the new timepoint to the template
            t = traditionalRegistrationThread(i, str(i).zfill(3), templateFn, timepointFns[i], outFn, outputDir, templatePrefix, prealign=prealign, regType=regType)
            myThreads.append(t)
            t.start()

    for t in myThreads:
        t.join()

    print(timepointFns)

    return registeredFns


def dagCorrection(timepoints, outputDir, transformPrefix, regType='nonlinear'):
    """
    Apply the dag motion correction algorithm to a timeseries image.
    Assumes that the first filename in the timepoints list specifies the
    template image.

    Inputs:
    - timepoints: list of filenames for each timepoint
    - outputDir: directory to write the output files to
    - transformPrefix: prefix for the transform files
    - regType: option argument to specify the registration type to use

    Outputs:
    - registeredFns: list of registered timepoint files

    Effects:
    - Writes each registered file to /path/dag-movement-correction/tmp/dag/
    """
    # get the template image filename
    templateFn = timepoints[0]
    # copy the template file to the registered directory
    shutil.copy(templateFn, outputDir)

    # set up list
    registeredFns = [outputDir+fn.split("/")[-1].split(".")[0]+'.nii.gz' for fn in timepoints]

    # register the first timepoint to the template
    registerToTemplate(templateFn, timepoints[1], registeredFns[1], outputDir, transformPrefix, initialize=False, regType=regType)

    # register the second timepoint to the template using the initialized transform
    registerToTemplate(templateFn, timepoints[2], registeredFns[2], outputDir, transformPrefix, initialize=True, initialRegFile=0, regType=regType)

    # for each subsequent image
    for i in range(3, len(timepoints)):
        # register the new timepoint to the template, using initialized transform
        registerToTemplate(templateFn, timepoints[i], registeredFns[i], outputDir, transformPrefix, initialize=True, initialRegFile=1, regType=regType)

    return registeredFns


#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Perform motion correction on time-series images.")
    # image filenames
    parser.add_argument('-i', '--inputFn', type=str, help='Full path to the name of the file to correct')
    parser.add_argument('-o', '--outputFn', type=str, help='The name of the file to save the correction to (within the directory containing the original file.')
    # which type of global volume registration framework
    parser.add_argument('-t', '--correctionType', type=str, help='Specify which type of correction to run. '
                        +'Options include: traditional or dag')
    # which type of registration
    parser.add_argument('-r', '--registrationType', type=str, help='Specify which type of registration to use: affine or nonlinear')

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

    if args.correctionType == 'traditional':
        """
        Traditional volume registration: align each image to the 
        first timepoint
        """
        # make the output directory
        outputDir = baseDir+'firstTimepointMatching/'
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        # copy the first image to the output directory
        shutil.copy(timepointFns[0], outputDir)
        # print(timepointFns[0])

        # register the images sequentially
        templateImg = timepointFns[0]
        registeredFns = volumeRegistration(templateImg, timepointFns, outputDir, baseDir, regType=args.registrationType)
        registeredFns.append(outputDir+'000.nii.gz')

    elif args.correctionType == 'dag':
        """
        DAG-based registration: Treat the image series as a DAG.
        - Treat the first volume as the template volume
        - Register the second volume to the first
        - Use the transformation generated by this registration to
          initialize the registration between the third volume and
          the first volume
        - Repeat the previous step for the remaining volumes in the
          image sequence
        """
        # make the output directory
        outputDir = baseDir+'dag/'
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        # load the template image
        img = load_image(timepointFns[0])
        coord = img.coordmap
        template = Image(img, coord)
        # save the template image in the tmp directory
        if not os.path.exists(tmpDir+"templates/"):
            os.mkdir(tmpDir+"templates/")
        save_image(template, tmpDir+"templates/dag_"+timepointFns[0].split('/')[-1])

        # set up the variable to indicate the location of the transform prefix
        if not os.path.exists(tmpDir+"prealignTransforms/"):
            os.mkdir(tmpDir+"prealignTransforms/")
        transformPrefix = tmpDir+"prealignTransforms/dag_"

        # register the images using dag correction
        registeredFns = dagCorrection(timepointFns, outputDir, transformPrefix, regType=args.registrationType)

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

    # Want to time how long it takes to register the image        
    startTime = time.time()
    subjFn, method = main()
    endTime = time.time() - startTime

    # Convert the amount of time taken from a float to a string
    totalDays = np.floor(endTime/24.0/60.0/60.0)
    endTime = endTime%(24.0*60.0*60.0)
    totalHours = np.floor(endTime/60.0/60.0)
    endTime = endTime%(60.0*60.0)
    totalMins = np.floor(endTime/60.0)
    totalSecs = endTime%60.0
    # Notify the user of the runtime
    print("Total run time:",totalDays,"days,",totalHours,"hours,",totalMins,"minutes,",totalSecs)

    # write the time to a file
    subj = subjFn.split("/")[-2]
    baseDir = subjFn.split(subj)[0]
    fn = baseDir+"timeToRun.csv"
    timeLine = subj+", "+ method+", "+ str(totalDays).zfill(2) + ":" + str(totalHours).zfill(2) + ":" + str(totalMins).zfill(2) + ":" + str(totalSecs).zfill(2) + "\n"
    # if the file aready exists
    if not os.path.isfile(fn):
        # open the file in write
        with open(fn, "w") as file:
            headerLine = "Subject, Method, RunTime (DD:HH:MM:SS)\n"
            # write the header line
            file.write(headerLine)
            # write the time line
            file.write(timeLine)

    else:
        with open(fn, "a+") as file:
            # write the time line
            file.write(timeLine)

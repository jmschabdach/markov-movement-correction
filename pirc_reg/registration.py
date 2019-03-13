from __future__ import print_function
import numpy as np
import os
import argparse
import shutil
import sys
import time

# for the registration
from nipype.interfaces.ants import Registration, ApplyTransforms
from nipype.algorithms.metrics import Similarity

# threading
import multiprocess

from image import ImageSequence

class RegistrationProcess():
    """
    Class definition for the registration process.

    Atributes:
    - framework: string either "traditional" or "dag"
    - reg_params: dictionary of registration parameters 
                  determined by affine or nonlinear
    - orig_img: ImageSequence object to register
    - reference_img: the string of the filename of the volume to use as the reference image
    - reg_img: ImageSequence object of registered orig_img
    """
    def __init__(framework, dof, imgSeq):
        """
        Initalize new instance of a RegistrationProcess object

        Arguments:
        - framework: string either "traditional" or "dag"
        - dof: degrees of freedom of the registration, either "affine" or "nonlinear"
        - imgSeq: ImageSequence to register

        Effects:
        - set the imgFn and/or seqFns attributes

        Returns:
        - None
        """
        self.framework = framework
        self.orig_img = imgSeq
        self.reference_img = ""
        self.reg_params = {}

        # Specify certain parameters for the nonlinear/['SyN'] registration
        if dof == 'nonlinear':
            self.reg_params["transforms"] = ['Affine', 'SyN']
            self.reg_params["transform_parameters"] = [(2.0,),(0.25, 3.0, 0.0)]
            self.reg_params["number_of_iterations"] = [[1500, 200], [100, 50, 30]] 
            self.reg_params["metric"] = ['CC']*2
            self.reg_params["metric_weight"] = [1]*2
            self.reg_params["radius_or_number_of_bins"] = [5]*2
            self.reg_params["convergence_threshold"] = [1.e-8, 1.e-9]
            self.reg_params["convergence_window_size"] = [20]*2
            self.reg_params["smoothing_sigmas"] = [[1,0],[2,1,0]]
            self.reg_params["sigma_units"] = ['vox']*2
            self.reg_params["shrink_factors"] = [[2,1],[3,2,1]]
            self.reg_params["use_estimate_learning_rate_once"] = [True,True]
            self.reg_params["use_histogram_matching"] = [True, True] # This is the default value, but specify it anyway

        # Specify certain parameters for the affine/['Affine'] registration
        elif dof == 'affine':
            self.reg_params["transforms"] = ['Affine']
            self.reg_params["transform_parameters"] = [(2.0,)]
            self.reg_params["number_of_iterations"] = [[1500, 200]] 
            self.reg_params["metric"] = ['CC'] 
            self.reg_params["metric_weight"] = [1]
            self.reg_params["radius_or_number_of_bins"] = [5] 
            self.reg_params["convergence_threshold"] = [1.e-8]
            self.reg_params["convergence_window_size"] = [20]
            self.reg_params["smoothing_sigmas"] = [[1,0]]
            self.reg_params["sigma_units"] = ['vox']
            self.reg_params["shrink_factors"] = [[2,1]]
            self.reg_params["use_estimate_learning_rate_once"] = [True]
            self.reg_params["use_histogram_matching"] = [True]

        else:
            print("The supported options for the dof argument are 'nonlinear' and 'affine'.")
            print("Please change to one of these options and retry.")
            exit(-1)

        # get the path of the original image
        if not self.orig_img.getImageFilename() == "":
            path = os.path.dirname(self.orig_image.getImageFilename())
        elif not len(self.orig_img.getSequenceFilenames()) == 0:
            path = os.path.dirname(os.path.dirname(self.orig_image.getSequenceFilenames()[0]))

        # create the new directory for the registered image
        regSequencePath = os.path.join(path, self.framework)
        if not os.path.isdir(regSequencePath):
            os.path.mkdir(regSequencePath)

        # make a new ImageSequence which takes a directory as an input
        self.reg_img = ImageSequence(seqDir=regSequencePath)


    def runRegistration():
        """
        Run the entire registration pipeline for the ImageSequence

        Inputs:
        - none

        Effects:
        - Registers all volumes in the ImageSequence to the first volume

        Returns:
        - none
        """
        # split the ImageSequence into timepoints
        if len(self.image_seq.getSequence()) == 0:
            self.image_seq.expandImageToSequence()
        # save the first image filename as the refrerence_img attribute
        self.reference_img = self.image_seq.getSequenceFns()[0]
        # copy the template file to the registered directory
        # JENNA CHECK THIS BIT
        shutil.copy(self.reference_img, self.reg_img.getSequencePath)
        newFn = os.path.join(self.reg_img.getSequencePath, os.path.basename(self.reference_img))
        self.reg_img.addToSequence(newFn)

        if self.framework == "traditional":
          self.traditionalRegistration()
        elif self.framework == "dag":
          self.dagCorrection()
        # print("Registration pipeline complete")


    def traditionalRegistration(self):
        """
        Register each image frame to the template image. (all to one)

        Inputs:
        - templateFn: the filename of the template image
        - timepointFns: list of filenames for each timepoint
        - outputDir: directory to write the output files to

        Outputs:
        - registeredFns: list of registered timepoint files

        Effects:
        - Writes each registered file to /path/dag-movement-correction/tmp/registered/
        """

        # set up lists
        registeredFns = []
        myThreads = []
        # for each subsequent image
        for i in range(1, len(self.orig_img.getSequenceFilenames()):
            # set the output filename
            outFn = os.path.join(self.reg_img.getSequencePath(), str(i).zfill(3)+'.nii.gz'
            self.reg_img.addToSequence(outFn)
            templatePath = os.path.join(os.path.dirname(self.reg_img.getSequencePath()), 'tmp/')
            # start a thread to register the new timepoint to the template
            parameters = [self.reference_img,
                          self.orig_img.getSequenceFilenames()[0],
                          outFn, templatePath+'output_']

            t = multiprocessing.Process(target=self.registerToTemplate, args=parameters)
            myThreads.append(t)
            t.start()

        for t in myThreads:
            t.join()

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

    def registerToTemplate(fixedImgFn, movingImgFn, outFn, transformPrefix, initialize=False, initialRegFile=0):
        """
        Register 2 images taken at different timepoints.

        Inputs:
        - fixedImgFn: filename of the fixed image (should be the template image)
        - movingImgFn: filename of the moving image (should be the Jn image)
        - outFn: name of the file to write the transformed image to.
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

        reg.inputs.transforms = self.reg_params["transforms"]
        reg.inputs.transform_parameters = self.reg_params["transform_parameters"]
        reg.inputs.number_of_iterations = self.reg_params["number_of_iterations"]
        reg.inputs.metric = self.reg_params["metric"]
        reg.inputs.metric_weight = self.reg_params["metric_weight"]
        reg.inputs.radius_or_number_of_bins = self.reg_params["radius_or_number_of_bins"]
        reg.inputs.convergence_threshold = self.reg_params["convergence_threshold"]
        reg.inputs.convergence_window_size = self.reg_params["convergence_window_size"]
        reg.inputs.smoothing_sigmas = self.reg_params["smoothing_sigmas"]
        reg.inputs.sigma_units = self.reg_params["sigma_units"]
        reg.inputs.shrink_factors = self.reg_params["shrink_factors"]
        reg.inputs.use_estimate_learning_rate_once = self.reg_params["use_estimate_learning_rate_once"]
        reg.inputs.use_histogram_matching = self.reg_params["use_histogram_matching"]

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
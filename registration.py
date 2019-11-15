from boldli import ImageManipulatingLibrary as mil
import numpy as np
from nipype.interfaces.ants import Registration, ApplyTransforms
from nipype.algorithms.metrics import Similarity
import os

##
# Expand an image sequence stored as a .nii.gz file into
# a collection of .nii.gz images where each frame is its
# own .nii.gz file.
#
# @param imgFn The time series image's filename
# @param outDir The directory in which a directory holding the collection of timepoint files
#
# @returns filenames List of strings specifying paths to timepoint files
def expandTimepoints(imgFn, outDir):
    # Load the image
    sequence, coords = mil.loadBOLD(imgFn)

    # Pull out the first volume from the sequence (timepoint 0)
    vol = mil.isolateVolume(sequence, volNum=0)

    # Save the first volume as timepoints/000.nii.gz
    volImg = mil.convertArrayToImage([vol], coords)
    fn = outDir+str(0).zfill(3)+'.nii.gz'
    mil.saveBOLD(volImg, fn)

    # Add the filename for the first volume to a new list of filenames for the timepoints
    filenames = [fn]

    # For the rest of the image volumes in the sequence
    for i in range(1, sequence.shape[-1], 1):

        # Pull out the next volume from the template and save it
        vol = mil.isolateVolume(sequence, volNum=i)

        # Save the first volume as timepoints/000.nii.gz
        volImg = mil.convertArrayToImage([vol], coords)
        fn = outDir+str(i).zfill(3)+'.nii.gz'
        mil.saveBOLD(volImg, fn)

        # add the name of the image to the list of filenames
        filenames.append(fn)

    return filenames

##
# Register a pair of images (previously registerToTemplate)
#
# Effects: save a copy of the registered image and the registration parameters
#
# @param fixedImgFn The filename of the fixed image as a string
# @param movinImgFn The filename of the moving image as a string
# @param regImgOutFn The filename as a string specifying where to save the registered moving image
# @param transformPrefix 
# @param initialize Optional parameter to specify the location of the transform matrix from the previous registration
# @param regType Optional parameter to specify the type of registration to use (affine ['Affine'] or nonlinear ['Syn']) Default: nonlinear
def registerVolumes(fixedImgFn, movinImgFn, regImgOutFn, transformPrefix, initialize=None, regtype='nonlinear'):

    # if the registered image already exists, skip the registration
    if os.path.exists(regImgOutFn):
        print("File", regImgOutFn, "already exists. Skipping registration.")
        return
    
    # Registration set up: for both Affine and SyN transforms
    reg = Registration()
    reg.inputs.fixed_image = fixedImgFn
    reg.inputs.moving_image = movinImgFn
    reg.inputs.output_transform_prefix = transformPrefix  # what does this line do?
    reg.inputs.interpolation = 'NearestNeighbor'
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = False  # what does this line do?
    reg.inputs.collapse_output_transforms = False
    reg.inputs.initialize_transforms_per_stage = False
    reg.inputs.num_threads = 100
    reg.inputs.output_warped_image = regImgOutFn

    # Registration set up: Specify certain parameters for the Affine registration step
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

    # Registration set up: SyN transforms only -- NEEDS TO BE CHECKED
    if regtype == 'nonlinear':
        reg.inputs.transforms.append('SyN')
        reg.inputs.transform_parameters.append((0.25, 3.0, 0.0))
        reg.inputs.number_of_iterations.append([100, 50, 30])
        reg.inputs.metric.append('CC')
        reg.inputs.metric_weight.append(1)
        reg.inputs.radius_or_number_of_bins.append(5)
        reg.inputs.convergence_threshold.append(1.e-9)
        reg.inputs.convergence_window_size.append(20)
        reg.inputs.smoothing_sigmas.append([2,1,0])
        reg.inputs.sigma_units.append('vox')
        reg.inputs.shrink_factors.append([3,2,1])
        reg.inputs.use_estimate_learning_rate_once.append(True)
        reg.inputs.use_histogram_matching.append(True) # This is the default value, but specify it anyway

    # If the registration is initialized, set a few more parameters
    if initialize is not None:
        reg.inputs.initial_moving_transform = initialize
        reg.inputs.invert_initial_moving_transform = False

    # Keep the user updated with the status of the registration
    print("Starting", regtype, "registration for", regImgOutFn)
    # Run the registration
    reg.run()
    # Keep the user updated with the status of the registration
    print("Finished", regtype, "registration for", regImgOutFn)

##
# Combine the images in the list of registered image volumes into a single image sequence file and save it
#
# @param registeredFns List of filenames (strings) for registered image files
# @param coords AffineTransform object form nipy specifying the coordinate map for the image sequence
# @param outFn String specifying where the registered image sequence should be saved
def stackNiftis(registeredFns, coords, outFn):

    print("Inside stackNiftis. Printing list of registered filenames.")
    print(registeredFns)

    imgs = []

    # Load all of the images
    for imgFn in sorted(registeredFns):
        # load a single image - use mil here
        img, noCoords = mil.loadBOLD(imgFn)
        # get the contents of the image
        vol = mil.isolateVolume(img)
        imgs.append(vol)

    # Conver the list of images to an array
    imgSeq = np.stack(imgs, axis=-1)
    print(imgSeq.shape)
    # Convert array of images to an Image object
    imgSequence = mil.convertArrayToImage(imgSeq, coords)
    mil.saveBOLD(imgSequence, outFn)
    print('Individual image volume files merged to',outFn)

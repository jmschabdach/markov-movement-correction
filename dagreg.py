"""
Jenna Schabdach 2019

Description

Useage:

"""
import shutil
import registration as reg

##
# Perform volume registration using the DAG-based method on a volumetric image sequence
# Assumes the first filename in the timepoints parameter specifies the template image
#
# @param timepoints List of filenames for each timepoint
# @param outputDir The directory where the registered images will be saved as a str
# @param transformPrefix The directory where the transforms will be saved as a str
# @param regType The type of transformations to use (either "affine" or "nonlinear")
#
# @returns registeredFns A list of filenames of the registered files
def dagRegistration(timepoints, outputDir, transformPrefix, transformType='nonlinear'):
    # get the template image filename
    templateFn = timepoints[0]
    # copy the template file to the registered directory
    shutil.copy(templateFn, outputDir)

    # Create list of filenames for the registered images
    registeredFns = []
    transformFns = []

    # Generate first registered image filename - Should generalize it better
    imgFn = outputDir+"1".zfill(3)+".nii.gz"
    transformFn = transformPrefix+"_"+"1".zfill(3)+".nii.gz"

    # Save the image filename and the transform filename
    registeredFns.append(imgFn)
    transformFns.append(transformFn)

    # register the first timepoint to the template
    reg.registerVolumes(templateFn, timepoints[1], imgFn, 
                        transformFn, initialize=None,
                        regtype=transformType)

    for i in range(2, len(timepoints)):
        # Generate next filenames
        imgFn = outputDir+str.zfill(i)+".nii.gz"
        transformFn = transformPrefix+"_"+str(i).zfill(3)+"_"

        # Save the image filename and the transform filename
        registeredFns.append(imgFn)
        transformFns.append(transformFn)

        # Register the ith timepoint to the template using the initialized transform
        reg.registerVolumes(templateFn, timepoints[i], imgFn, 
                            transformFn, initialize=transformFns[-1],
                            regtype=transformType)

    return registeredFns

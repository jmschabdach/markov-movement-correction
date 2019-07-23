"""
Jenna Schabdach 2019

Description

Useage:

"""

import threading
import registration as reg

class traditionalRegistrationThread(threading.Thread):
    """
    Implementation of the threading class.

    Purpose: parallelize the traditional registration process
    """
    def __init__(self, threadId, name, templateFn, timepointFn, outputFn, outputDir, transformPrefix, regType='nonlinear'):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = name
        self.templateFn = templateFn
        self.timepointFn = timepointFn
        self.outputFn = outputFn
        self.outputDir = outputDir
        self.transformPrefix = transformPrefix
        self.regType = regType

    def run(self):
        print("Starting traditional registration for", self.name)
        reg.regsisterVolumes(self.templateFn, self.timepointFn, self.outputFn, self.transformPrefix, regType=self.regType)
        print("Finished traditional registration for", self.name)


##
# Register each image volume directly to the reference image volume using the traditional framework
#
# @param referenceFn The filename of the reference image volume
# @param timepoints The list of filenames for each timepoint
# @param outputDir The directory to write the registered images as a str
# @param transformDir The directory to write the transform parameters to as a str
# @param regType The type of transformations to use (either "affine" or "nonlinear")
#
# @returns registeredFns A list of filenames of the registered files
def volumeRegistration(referenceFn, timepoints, outputDir, transformDir, regType='nonlinear'):
    # Create lists of registered images, transform parameters, and threads
    registeredFns = []
    transformFns = []
    myThreads = []
    # for each subsequent image
    for i in range(len(timepointFns)):
        if timepointFns[i] == templateFn:
            # copy the template file into the output directory
            shutil.copy(referenceFn, outputDir)
            print("FOUND THE TEMPLATE FILE")
        else:
            # set the output filename
            outFn = outputDir+str(i).zfill(3)+'.nii.gz'
            registeredFns.append(outFn)
            templateFns = transformDir+str.zfill(i)+"_" #+'tmp/output_'
            # start a thread to register the new timepoint to the template
            t = traditionalRegistrationThread(i, str(i).zfill(3), referenceFn,
                                              timepointFns[i], outFn, outputDir,
                                              transformDir+"trad_"+str(i).zfill(3)+"_", regType=regType)
            myThreads.append(t)
            t.start()


    for t in myThreads:
        t.join()

    return registeredFns

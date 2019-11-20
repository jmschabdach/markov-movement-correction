"""
Jenna Schabdach 2019

Description

Useage:

"""

import threading
import shutil
import tempfile
import os
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
        reg.registerVolumes(self.templateFn, self.timepointFn, self.outputFn, self.transformPrefix, regtype=self.regType)
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
def volumeRegistration(referenceFn, timepointFns, outputDir, transformDir, regType='nonlinear'):
    # Create lists of registered images, transform parameters, and threads
    registeredFns = []
    transformFns = []
    myThreads = []

    maxThreads = 20
    count = 0
    # for each subsequent image
    for i in range(len(timepointFns)):
        outFn = outputDir+str(i).zfill(3)+'.nii.gz'
        registeredFns.append(outFn)

        if timepointFns[i] == referenceFn:
            # copy the template file into the output directory
            shutil.copy(referenceFn, outputDir)
            print("FOUND THE TEMPLATE FILE")
        else:
            # set the output filename
            #templateFns = transformDir+str(i).zfill(3)+"_" #+'tmp/output_'
            # make a new copy of the reference volume - trying to solve errors on H2P
            newDir = tempfile.mkdtemp()
            shutil.copy(referenceFn, newDir)
            newReferenceFn = os.path.join(newDir, referenceFn)
            # start a thread to register the new timepoint to the template
            t = traditionalRegistrationThread(i, str(i).zfill(3), newReferenceFn,
                                              timepointFns[i], outFn, outputDir,
                                              transformDir+"trad_"+str(i).zfill(3)+"_", regType=regType)
            myThreads.append(t)
            t.start()
            count += 1

        if (count >= maxThreads):
            for t in myThreads:
                t.join()

            print("Max number of threads reached:", len(myThreads))
            myThreads = []
            print("Reset the list of threads:", len(myThreads))
            count = 0

        elif i == (len(timepointFns) - 1):
            print("Last set of threads started.")
            for t in myThreads:
                t.join()
            print("Last set of threads completed.")

    return registeredFns

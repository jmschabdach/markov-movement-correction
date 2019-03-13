"""
Jenna Schabdach 2019



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

# for saving the registered file
from nipype.interfaces import dcmstack

class ImageSequence():
    """
    Class definition for the image sequence

    Attributes:
    - imgFn: the filename of the image sequence file as a single string
    - seqFns: the list of filenames of the images in the sequence (list of strings)
    - img: 
    - coords: coordinate frame of the image
    - sequence: list of images in the file sequence
    """
    def __init__(self, imgFn="", seqDir=""):
        """
        Initalize new instance of Image Sequence object

        Arguments:
        - imgFn: the filename of the image sequence file as a single string
        - seqDir: the path to the directory of the set of image sequence files as a single string

        Effects:
        - set the imgFn and/or seqFns attributes

        Returns:
        - None
        """
        self.imgFn = imgFn
        self.seqFns = []
        self.img = None
        self.coords = None
        self.sequence = []

        if seqDir is not "":
            # add all files ending in .nii or .nii.gz that are file to the list of sequence files
            # check that the directory exists
            if not os.path.exists(seqDir):
                raise IOError('Error: the specified directory does not exist')
            # check that directory contains .nii or .nii.gz images
            self.seqFns = [os.path.join(seqDir, f) for f in os.listdir(seqDir) if os.path.isfile(os.path.join(seqDir, f)) and (f.endswith('.nii.gz') or f.endswith('.nii'))]
            self.seqFns = sorted(self.seqFns)

    def loadImage(self):
        """
        Load the image sequence file

        Arguments:
        - None

        Effects:
        - Load the image file specified in the imgFn attribute

        Returns:
        - None
        """
        self.img = load_image(self.imgFn)
        self.coords = self.img.coordmap

    def loadSequence(self):
        """
        Load the individual files in the image

        Arguments:
        - None

        Effects:
        - Load the list of image files specified in the seqFns attribute

        Returns:
        - None
        """
        imgs = []
        for fn in sorted(self.seqFns):
            img = load_image(fn)
            self.coords = img.coordmap
            if len(img.get_data().shape) == 4:
                imgs.append(np.squeeze(img.get_data()))
            else:
                imgs.append(img.get_data())
        self.sequence = imgs
        
    def saveImage(self, outFn=""):
        """
        Save the image sequence to the file specified by imgFn

        Arguments:
        - None

        Effects:
        - Create or overwrite the image file specified in the imgFn attribute

        Returns:
        - None
        """
        if outFn == "":
            outFn = self.imgFn
        # save the image sequence as a single image
        registeredImg = Image(self.img, self.coords)
        save_image(registeredImg, outFn)
        self.imgFn = outFn
        print('ImageSequence sequence files merged to', outFn)

    def getSequence(self):
        """
        Return the list of images contained within the image sequence

        Arguments:
        - None

        Effects:
        - None

        Returns:
        - the list of images within the image sequence stored in the sequence attribute
        """
        return self.sequence

    def getSequenceFilenames(self):
        """
        Return the list filenames for the images contained within the image sequence

        Arguments:
        - None

        Effects:
        - None

        Returns:
        - the list of images within the image sequence stored in the sequence attribute
        """
        return self.seqFns

    def getCoordinates(self):
        """
        Return the coordinates of the image
        """
        return self.coords

    def getImage(self):
        """
        Return the variable containing the entire image sequence

        Arguments:
        - None

        Effects:
        - None

        Returns:
        - the entire image sequence stored in the image attribute
        """
        return self.img

    def getImageFilename(self):
        """
        Return the filename for the image
        """
        return self.imgFn

    def setCoordinates(self, coordinates):
        """
        Set the coordinates of the image
        """
        self.coords = coordinates

    def addToSequence(self, fn):
        """
        Add a filename to the list of filename strings representing the images volumes
        in the image sequence

        Arguments:
        - fn: the filename to add to the list of image volume filenames as a string

        Effects:
        - None

        Returns:
        - None
        """
        self.seqFns.append(fn)

    def expandImageToSequence(self, dirName='timepoints'): # previously expandTimepoints
        """
        Expand an image sequence stored as a .nii.gz file into a collection of 
        .nii.gz images (where each frame is its own .nii.gz file)

        Arguments:
        - dirName: name of the directory to make and expand the image into

        Effects:
        - Save the image volume at each timepoint of the image to a new file

        Returns:
        - filenames: list of filenames
        """
        outDir = os.path.join(os.path.dirname(self.imgFn), dirName)
        if not os.path.exists(outDir):
            os.mkdir(outDir)

        print("Image has length:", self.img.get_data().shape[3])

        # for the remaining images
        for i in range(self.img.get_data().shape[3]):
            # pull out the image and save it
            tmp = self.img[:,:,:,i].get_data()[:,:,:,None]
            tmp_img = Image(tmp, self.coords)
            outFn = str(i).zfill(3)+'.nii'
            save_image(tmp_img, os.path.join(outDir, outFn))
            # add the name of the image to the list of filenames
            self.addToSequence(os.path.join(outDir, outFn))

    def condenseSequenceToImage(self): # previously stackNifitis
        """
        Combine the list of registered image frames into a single image sequence file.

        Arguments:
        - origFn: filename of the original image file
        - registeredFns: list of filenames for the registered timepoint images
        - outFn: name of the file to write the combined image to

        Returns:
        - Nothing

        Effect:
        - Combine the registered timepoint images into a single file
        """
        self.loadSequence()
        # stack the image data
        self.img = np.stack(self.sequence, axis=-1)
       
def compareImageSequences(seq1, seq2): # leave like this or change to use ImageSequence objects?
    """
    Compare two image sequences

    Arguments:
    - seq1: ImageSequence object
    - seq2: ImageSequence object

    Effects:
    - Prints whether or not the two images are equal; 
        Whether the filenames match or not does not impact
        the output.

    Returns:
    - areSame: boolean flag indicating whether the two images are equal
    """
    # load the two image sequences
    if len(seq1.getSequence()) == 0:
        if not len(seq1.getSequenceFilenames()) == 0:
            print("Loading image sequence for seq1")
            seq1.loadSequence()
            print(seq1.getSequenceFilenames()[0])
        elif not seq1.getImageFilename() == "":
            print("Loading image file for seq1 and expanding into sequence")
            seq1.loadImage()
            seq1.expandImageToSequence()
            seq1.loadSequence()

    if len(seq2.getSequence()) == 0:
        if not len(seq2.getSequenceFilenames()) == 0:
            print("Loading image sequence for seq2")
            seq2.loadSequence()
            print(seq2.getSequenceFilenames()[0])
        elif not seq2.getImageFilename() == "":
            print("Loading image file for seq2 and expanding into sequence")
            seq2.loadImage()
            seq2.expandImageToSequence()
            seq2.loadSequence()

    img1Data = seq1.getSequence()
    img2Data = seq2.getSequence()

    areSame = True
    # areExact = True
    for vol1, vol2 in zip(img1Data, img2Data):
        # if not np.array_equal(vol1, vol2):
        #     areExact = False
        if not np.allclose(vol1, vol2):
            areSame = False

    if areSame:
        print("Images are the same, except for possible minor rounding errors.")
    else:
        print("Images are different")

    # print("areExact:", areExact)

    return areSame


if __name__ == "__main__":
    
    print("Hello world")

    # test creating a new ImageSequence using a filename
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    fn1 = os.path.join(os.path.join(dir_path, 'test_image'), 'BOLD.nii')
    imgSeq1 = ImageSequence(imgFn=fn1)
    print("Created an ImageSequence from the filename of an image")
    # test loadImage
    imgSeq1.loadImage()
    print("Loaded the image of an ImageSequence")
    # test getImage
    img = imgSeq1.getImage()
    print("The image of ImageSequence1 has the shape:", img.shape)
    # test expandImageToSequence
    # test addToSequence - tested as part of expandImageToSequence()
    imgSeq1.expandImageToSequence(dirName='foobar/')
    print("The ImageSequence1 was expanded into the subdirectory 'foobar'")
    fn3 = os.path.join(os.path.join(dir_path, 'test_image'), 'BOLD_2.nii')
    imgSeq1.saveImage(fn3)

    # test creating a new ImageSequence using a directory of image volumes
    dir2 = os.path.join(os.path.join(dir_path, 'test_image'), 'foobar/')
    imgSeq2 = ImageSequence(seqDir=dir2)
    print("Created a second ImageSequence from a directory containing a sequence")
    # test loadSequence
    imgSeq2.loadSequence()
    print("Loaded the sequence images for ImageSequence2")
    # test getSequence
    print(len(imgSeq2.getSequence()))
    # test condenseSequenceToImage
    imgSeq2.condenseSequenceToImage()
    # test saveImage
    fn2 = os.path.join(os.path.join(dir_path, 'test_image'), 'condensed_image_test.nii')
    imgSeq2.saveImage(fn2)

    # test comparing two images that are the same
    imgSeq3 = ImageSequence(imgFn=fn1)
    imgSeq3.loadImage()

    # test compareImages
    compareImageSequences(imgSeq1, imgSeq1)
    compareImageSequences(imgSeq1, imgSeq2)
    compareImageSequences(imgSeq1, imgSeq3)
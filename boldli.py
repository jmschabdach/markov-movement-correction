from __future__ import print_function
import numpy as np
import os
from os import listdir
from os.path import isfile, join

# For loading/saving the images
from nipy.core.api import Image
from nipy import load_image, save_image
from nipype.interfaces import dcmstack


class ImageManipulatingLibrary:
    ##
    # Load the image
    #
    # @param fn The string specifying the path to the file
    #
    # @returns img The image sequence as an Image object
    # @returns coords The coordinate system for the image sequence
    def loadBOLD(fn):
        img = load_image(fn)
        coords = img.coordmap

        return img, coords

    ##
    # Isolate a single volume from the sequence
    #
    # @param seq The image sequence as an Image object
    # @param volNum An int specifying the volume to isolate
    #
    # @returns vol The isolated volume
    def isolateVolume(seq, volNum=0):
        # Check the shape of the image
        print(seq.shape)
        if len(seq.shape) == 4:
            # Pull out the volume of interest from the sequence
            vol = seq[:,:,:, volNum].get_data()[:,:,:, None]
            # Make sure that the volume only has 3 dimensions
            vol = np.squeeze(vol)
        elif len(seq.shape) == 3:
            vol = seq.get_data()
        else:
            print("The volume must have 3 or 4 dimensions. Given volume has shape:", seq.shape)

        return vol

    ##
    # Load the image sequence from a directory
    #
    # @param directory The string specifying the path to the directory
    # @returns seq The image
    def loadSequenceFromDir(directory):
        # Check to see if the directory exists
        if not os.path.exists(directory):
            raise IOError('Error: the specified directory does not exist')

        # Get the list of .nii/.nii.gz files in the directory
        files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f)) and (f.endswith('.nii.gz') or f.endswith('.nii'))];

        # For every file in the sorted list of files
        imgs = []
        for fn in sorted(files):
            img = load_image(fn)
            if len(img.get_data().shape) == 4:
                imgs.append(np.squeeze(img.get_data()))
            else: 
                imgs.append(img.get_data())

        # Stack the list of images into a numpy array
        stacked = np.stack(imgs, axis=-1)

        # Get the image coordinates
        coords = img.coordmap

        # Conver the numpy array into an image
        sequence = Image(stacked, coords)

        return sequence

    ##
    # Convert a 4D numpy array to an Image object
    #
    # @param seq The image sequence as a 4D numpy array
    # @param coords The coordinates for the image sequence
    #
    # @returns seqImg The sequence as an Image object
    def convertArrayToImage(seq, coords):
        # Condense the replicated sequence
        seqStack = np.stack(seq, axis=-1)

        # Convert the image sequence into an Image
        seqImg = Image(seqStack, coords)

        return seqImg

    ##
    # Compare the data and coordinate maps of two Images
    #
    # @param img1 First Image
    # @param img2 Second Image
    #
    # @returns True/False
    def compareImages(img1, img2):
        # Get the coordinate maps of the images
        coords1 = img1.coordmap
        coords2 = img2.coordmap

        # Get the data of the images
        data1 = img1.get_data()
        data2 = img2.get_data()

        # Compare the coordinate maps and data
        if coords1 == coords2 and data1.all() == data2.all():
            return True
        else:
            return False


    ## 
    # Save the standardized image sequence
    #
    # @param seq The standardized image sequence
    # @param fn The location to save the image sequence to
    def saveBOLD(seq, fn):
        save_image(seq, fn)


def main():
    print("Image Manipulating Library")    

if __name__ == "__main__":
    main()

import argparse
from boldli import ImageManipulatingLibrary as mil
import os
import sys

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str)
    parser.add_argument('-d', '--directory', type=str)

    # Parse the arguments
    args = parser.parse_args()
    seqFn = args.file
    seqPath = args.directory

    # Check to make sure the image and the directory exist
    if not os.path.exists(seqFn):
        raise IOError("Error: the specified image does not exist: "+ seqFn)

    if not os.path.exists(seqPath):
        raise IOError("Error: the specified directory does not exist: "+ seqPath)

    # Load the image from the file
    seqImg, coords = mil.loadBOLD(seqFn)
    # Load the image from the directory
    seqDir = mil.loadSequenceFromDir(seqPath)

    # Compare the two images
    status = mil.compareImages(seqImg, seqDir)

    if status:
        print("Images are equal")
        sys.exit(0)
    else:
        print("Images are different")
        sys.exit(1)

if __name__ == "__main__":
    main()

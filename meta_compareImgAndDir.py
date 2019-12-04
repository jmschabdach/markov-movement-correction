import argparse
from boldli import ImageManipulatingLibrary as mil
import sys

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str)
    parser.add_argument('-d', '--directory', type=str)

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    seqFn = args.file
    seqPath = args.directory

    # Load the image from the file
    seqImg, coords = mil.loadBOLD(seqFn)
    # Load the image from the directory
    seqDir = mil.loadSequenceFromDir(seqPath)

    # Compare the two images
    status = mil.compareImages(seqImg, seqDir)

    if status:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

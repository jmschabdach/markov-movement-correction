from __future__ import print_function
import argparse
from nipy import load_image, save_image
from nipy.core.api import Image
import os
import numpy as np
from os import listdir
from os.path import isfile, join

# set up the argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', help='Directory containing a series of timepoint .nii or .nii.gz images to combine into a single image',
                    required=True)
parser.add_argument('-o', '--out-image', help='Filename of the output image to write to (default directory is the input directory)',
                    default='', required=True)

# parse the args
args = parser.parse_args()
imgsDir = args.directory
outFn = args.out_image

# check that the directory exists
if not os.path.exists(imgsDir):
    raise IOError('Error: the specified directory does not exist')

# check that directory contains .nii or .nii.gz images
files = sorted([join(imgsDir, f) for f in listdir(imgsDir) if isfile(join(imgsDir, f)) and (f.endswith('.nii.gz') or f.endswith('.nii'))])

# make the output file name
if outFn == '':
    outFn = imgsDir+outFn

# Now stack the images
print("Loading image volumes...")
imgs = []
for fn in files:
    img = load_image(fn)
    if len(img.get_data().shape) == 4:
        imgs.append(np.squeeze(img.get_data()))
    else:
        imgs.append(img.get_data())

imgStack = np.stack(imgs, axis=-1)
print("Image volumes loaded")

# Compare the stacked image to the output image if it exists
if os.path.exists(outFn) and os.path.isfile(outFn):
    existingImg = load_image(outFn)

    if np.array_equal(existingImg.get_data(), imgStack):
        print("Old image has same info as new image")

    else:
        # get the coordinates
        coords = load_image(files[0]).coordmap
        
        # and save the stacked image
        registeredImg = Image(imgStack, coords)
        save_image(registeredImg, outFn)
        
        print('Nifti images merged to', outFn)

from __future__ import print_function
import argparse
from nipy import load_image, save_image
from nipy.core.api import Image
import os

# set up the argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help='Input image to be split. Must be a 3D time series (4 dimensions)', 
                    required=True)
parser.add_argument('-d', '--directory', help='Directory to write the split image to; default = ./tmp/',
                    default='./tmp/')

# parse the args
args = parser.parse_args()
imgFn = args.image
outDir = args.directory

# check the image for correct dimensionality
img = load_image(imgFn)
if len(img.get_data().shape) < 4 or img.get_data().shape[3] == 1:
    raise TypeError('Error: the image does not have the correct dimensionality')

# check the directory
if not os.path.exists(outDir):
    os.mkdir(outDir)
if not outDir[-1] == '/':
    # print(outDir)
    outDir += '/'
    # print(outDir)

# get base of image filename for use in output files
outImgBase = imgFn.split('/')[-1].split('.')[0]
# print('Base name for the output images:', outImgBase)

# split the image
coord = img.coordmap
for i in xrange(img.get_data().shape[3]):
    tmp = img[:,:,:,i].get_data()[:,:,:,None]
    tmpImg = Image(tmp, coord)
    outFn = outDir+str(i).zfill(3)+'.nii.gz'
    save_image(tmpImg, outFn)

# finished splitting the image
print('The image', imgFn,'was split into its timepoints.')
print('The timepoints can be found in:', outDir)

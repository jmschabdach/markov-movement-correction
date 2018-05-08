# DAGBasedRegistration
Volume registration for 4D images. Currently being tested with neonatal images. Future applications include pediatric and fetal images.

# Packages

- `numpy`: minimum version 1.12.1
- `nipy`
- `nipype`
- `os`
- `argparse`
- `shutil`
- `sys`
- `time`
- `threading`


I recommend using `conda` to manage all packages required by this project.

# Use

## Run a global volume registration process on an image

`python globalVolumeRegistration.py -i [full path to input image] -o [filename for registered image] -t [traditional or dat, type of global volume registration] -r [affine or nonlinear, registration type]`

- `-t`: indicates which type of registration to use. Current options are `dag` and `traditional`.
- The code generates a new directory with the same name as the input image. Within that directory, a subdirectory is generated with the name of the type correction specified. (i.e., the `dag` registered images are saved in a new directory `./<input 4D image path>/dag/`.)

## Calculate the similarities between the frames

`bash calculateSimilarity.sh <base directory> <template image>`: calculate the similarity between the 3D image at each timepoint and the template for the expanded 4D images in `<base directory>/dag/`, `<base directory>/traditional/`, and `<base directory>/timepoints/`. The `<base directory>/timepoints/` directory stores a collection of 3D images. Each image is a single timepoint from the original 4D input image.

# Directory Structure
 
subject-id/
&nbsp;&nbsp;|-- 4D-input-image-name.nii.gz (or name.nii)
&nbsp;&nbsp;|-- timepoints/  
&nbsp;&nbsp;|-- dag/  
&nbsp;&nbsp;|-- traditional/  
&nbsp;&nbsp;|-- tmp/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- currently used for transform matrix storage

# Current pipeline

- Run the movement correction for a single image using the desired options. 
- Evaluate the registration using `calculateSimilarity.sh`.
- Plot the similarities for each image using the workflow in `figureGeneration.ipynb`.
- Estimate the movement content in the original image using `fsl_motion_outliers` by MIT. (This utility produces a sparse matrix with ones in the columns where the timepoint has a significant amount of motion when compared to the other timepoints in the image.)

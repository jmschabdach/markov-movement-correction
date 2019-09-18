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

`python globalVolumeRegistration.py -i [full path to input image] 
-o [filename for registered image] 
-t [traditional or dat, type of global volume registration] 
-r [affine or nonlinear, registration type]`

- `-t`: indicates which type of registration to use. Current options are 
`dag` and `traditional`.
- The code generates a new directory with the same name as the input 
image. Within that directory, a subdirectory is generated with the name
of the type correction specified. (i.e., the `dag` registered images are
saved in a new directory `./<input 4D image path>/dag/`.)

## Calculate the similarities between all possible pairs of frames in each image

The script `calculateCorrelationMatrix.sh` calculates the correlation ratio
between every possible pair of frames in each imate - registered or 
unregistered. The correlation ratio measures the similarity of a pair of
images. A smaller correlation ratio indicates that the pair of images are 
more similar, while larger correlation ratios indicate less similarity 
between the images. 

After calculating each value, the script writes the value to a .csv file. 
When all values have been calculated, the .csv file is passed to a python
script. The python script removes extra characters from the end of each
line and saves the file.

Usage: 

`bash calculateCorrelationMatrix.sh [directory of image sequence files]
[output fn (.csv file)]`

## Estimate the amount of motion: framewise displacement and DVARS

To determine the usability of each image for clinical/research applications,
we calculate the framewise displacement and derivative of the variance of
the signal (DVARS) between each temporally neighboring pair of frames using
Power et at.'s definitions of these metrics. Power et al. also defined 
usability thresholds for each metric:

- Framewise displacement of < 0.2 mm between frames in 50% of frames
- DVARS change of < 25 voxel units between frames on a normalized scale of 
[0, 1000] units in 50% of frames

We use FSL's toolbox to calculate these metrics for each image sequence.
The script `powerMetrics.sh` calls `fsl_motion_outliers`, which takes 
a single image sequence file and calculates the framewise displacement
and DVARS metrics between each pair of temporally neighboring frames.
The script creates 4 .csv files: a pair of confound matrices and a pair
of list of metrics, where one file of each pair is for the framewise
displacement and the other is for the DVARS changes.

Usage:

`bash powerMetrics.sh [image sequence file name]`


# Directory Structure
 
subject-id/
&nbsp;&nbsp;+-- 4D-input-image-name.nii.gz (or name.nii)
&nbsp;&nbsp;+-- timepoints/  
&nbsp;&nbsp;+-- dag/  
&nbsp;&nbsp;+-- traditional/  
&nbsp;&nbsp;+-- tmp/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- currently used for transform matrix storage

# Current pipeline

- Run the movement correction for a single image using the desired options. 
- Evaluate the registration using `calculateSimilarity.sh`.
- Plot the similarities for each image using the workflow in `figureGeneration.ipynb`.
- Estimate the movement content in the original image using `fsl_motion_outliers` by MIT. (This utility produces a sparse matrix with ones in the columns where the timepoint has a significant amount of motion when compared to the other timepoints in the image.)


### Evaluating the registration methods

1. Run the command `bash calculateMetrics.sh` to calculate the correlation ratio matrices between every pair of frames in each image AND to calculate Power et al.'s framewise displacement and DVARS changes between every neighboring pair of frames.
2. Run the command `python condensePowerThresholdInfo.py --dir /path/to/dir/of/subjects/`. This file condenses the files generated in Step 2 into a pair of files containing the framewise displacement and the DVARS changes (displacementCounts.csv and intensityCounts.csv).
3. Open statisticalAnalyses/temporalKolmogorovSmirnovs.R. Change the paths at the top of the file to the paths to your `displacementCounts.csv` and `intensityCounts.csv` files. Running this file (with valid paths and no further changes) will produce 4 histograms of the framewise displacements and DVARS changes for the linear and nonlinear traditional and DAG based registrations.

=======
# References

1. 

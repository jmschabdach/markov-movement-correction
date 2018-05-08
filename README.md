# markov-movement-correction
Movement correction for 4D images. Currently being tested with neonatal images. Future applications include pediatric and fetal images.

# Requirements

Uses the `nipy` package. 

I recommend using `conda` to manage all packages required by this project.

# Use

`python hmmMovementCorrection.py -i <4D input image> -t <type of registration> -o <name of the file to save the registered 4D image to>`: expand the 4D image to a series of 3D images and register each timepoint to the image corresponding to image 0. 
- `-t`: indicates which type of registration to use. Current options are `hmm`, `sequential`, `bi-hmm`, and `stacking-hmm`.
- The code generates a new directory with the same name as the input image. Within that directory, a subdirectory is generated with the name of the type correction specified. (i.e., the `hmm` registered images are saved in a new directory `./<input 4D image name, sans extension>/hmm/`.)

`bash calculateSimilarity.sh <base directory> <template image>`: calculate the similarity between the 3D image at each timepoint and the template for the expanded 4D images in `<base directory>/hmm/`, `<base directory>/sequential/`, `<base directory>/bi-hmm/`, `<base directory>/stacking-hmm/`, and `<base directory>/timepoints/`. The `<base directory>/timepoints/` directory stores a collection of 3D images. Each image is a single timepoint from the original 4D input image.

# Directory Structure
 
subject-id/
&nbsp;&nbsp;|-- 4D-input-image-name.nii.gz 
&nbsp;&nbsp;|-- timepoints/  
&nbsp;&nbsp;|-- hmm/  
&nbsp;&nbsp;|-- stacking-hmm/  
&nbsp;&nbsp;|-- sequential/  
&nbsp;&nbsp;|-- tmp/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- currently used for transform matrix storage

# Current pipeline

- Run the movement correction for a single image using the desired options. 
- Evaluate the registration using `calculateSimilarity.sh`.
- Plot the similarities for each image using the workflow in `figureGeneration.ipynb`.
- Estimate the movement content in the original image using `fsl_motion_outliers` by MIT. (This utility produces a sparse matrix with ones in the columns where the timepoint has a significant amount of motion when compared to the other timepoints in the image.)

### Evaluating the registration methods

1. Run the command `bash calculateMetrics.sh` to calculate the correlation ratio matrices between every pair of frames in each image AND to calculate Power et al.'s framewise displacement and DVARS changes between every neighboring pair of frames.
2. Run the command `python condensePowerThresholdInfo.py --dir /path/to/dir/of/subjects/`. This file condenses the files generated in Step 2 into a pair of files containing the framewise displacement and the DVARS changes (displacementCounts.csv and intensityCounts.csv).
3. Open statisticalAnalyses/temporalKolmogorovSmirnovs.R. Change the paths at the top of the file to the paths to your `displacementCounts.csv` and `intensityCounts.csv` files. Running this file (with valid paths and no further changes) will produce 4 histograms of the framewise displacements and DVARS changes for the linear and nonlinear traditional and DAG based registrations.

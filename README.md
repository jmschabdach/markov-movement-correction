# markov-movement-correction
Movement correction for 4D images. Currently being tested with neonatal images. Future applications include pediatric and fetal images.

# Use

`python markov_movement_correction.py -i <input 4D image> -t <type of registration> -o <name of the file to save the registered 4D image to>`: expand the 4D image to a series of 3D images and register each timepoint to the image corresponding to image 0. 
- `-t`: indicates which type of registration to use. Current options are `markov` and `non-markov`. The `markov` registered images are saved in a new directory `./tmp/markov/` while the `non-markov` registered images are saved in new directory `./tmp/registered/`

`bash calculateSimilarity.sh`: calculate the similarity between the 3D image at each timepoint and the template (image at timepoint 0) for the expanded 4D images in `./tmp/markov/`, `./tmp/registered/`, and `./tmp/timepoints/	

# Current pipeline

- Run the movement correction for a single image using both the markov and the non-markov type options
- Evaluate the registration using `calculateSimilarity.sh`
- Plot the similarities for each image using the workflow in `figureGeneration.ipynb`
- Estimate the movement content in the original image using `fsl_motion_outliers` by MIT.

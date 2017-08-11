#!/bin/bash

# TODO: generalize the filename out

# set up any vars here
BASE='/home/jms565/Research/CHP-PIRC/markov-movement-correction'
TEMPLATE="$BASE/timepoints/000.nii.gz"
ORIG="$BASE/0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001.nii.gz"

# # clear the 0003_MR1... dir
# rm -rf "$BASE/0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001/"

# run the template matching registration
python hmmMovementCorrection.py -i $ORIG -o "$BASE/0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001/template_matching_correction.nii.gz" -t template

# run the first timepoint matching registration
python hmmMovementCorrection.py -i $ORIG -o "$BASE/0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001/first_timepoint_matching_correction.nii.gz" -t first-timepoint

# run the sequential registration
python hmmMovementCorrection.py -i $ORIG -o "$BASE/0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001/sequential_correction.nii.gz" -t sequential

# run the regular HMM registration
python hmmMovementCorrection.py -i $ORIG -o "$BASE/0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001/hmm_correction.nii.gz" -t hmm


# run the similarity calculations
# bash calculateSimilarity.sh "$BASE/0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001/" $TEMPLATE

echo "Finished running 4 registration algorithms. Go run the metric calculations."
# note: the metric calculation for the template matching algorithm needs to use the template name specified at $BASE/templateMatching-templateName.txt
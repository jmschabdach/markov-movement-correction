#!/bin/bash

# TODO: generalize the filename out

# set up any vars here
# BASE='/home/jms565/Research/CHP-PIRC/markov-movement-correction/Controls/'
# BASE='/home/jms565/Research/CHP-PIRC/markov-movement-correction/analysisII/Pittsburgh/'
# BASE='/home/jenna/Research/CHP-PIRC/markov-movement-correction/data/analysisII/Pittsburgh'
BASE='/home/jms565/Research/CHP-PIRC/markov-movement-correction/data/analysisII/Pittsburgh'
# BASE='/home/pirc/processing/Motion_Correction/Controls/'
# TEMPLATE="$BASE/timepoints/000.nii.gz"
# ORIG="0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001.nii.gz"


for i in "$BASE"/* ; do
    if [ -d "$i" ] ; then
        SUB_ORIG="$i/BOLD.nii"
        if [ -f "$SUB_ORIG" ]; then
            bash runAndNotify.sh python hmmMovementCorrection.py -i $SUB_ORIG -o corrected_firstTimepoint.nii.gz -t first-timepoint
        elif [ -f "$SUB_ORIG.gz" ]; then
            SUB_ORIG="$i/BOLD.nii.gz"
            bash runAndNotify.sh python hmmMovementCorrection.py -i $SUB_ORIG -o corrected_firstTimepoint.nii.gz -t first-timepoint
        else
            echo 'The specified file does not exist'
        fi
    fi
done

# # run the similarity calculations
# # bash calculateSimilarity.sh "$BASE/0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001/" $TEMPLATE

# echo "Finished running 4 registration algorithms. Go run the metric calculations."
# note: the metric calculation for the template matching algorithm needs to use the template name specified at $BASE/templateMatching-templateName.txt

#!/bin/bash

# Calculate all metrics associated with the motion correction analysis
# Updated analysis: 11/18/2017
# Usage: bash calculateMetrics.sh

#BASE="/home/jenna/Research/CHP-PIRC/markov-movement-correction/data/NonlinearControls-MI/"
#BASE="/home/jenna/Research/CHP-PIRC/markov-movement-correction/data/Controls/"
BASE=$1

# Just in case, remove all existing unzipped volumes and re-unzip the BOLDs
for i in "$BASE"/* ; do
    if [ -d "$i" ] ; then
        echo "$i"
        ORIG_DIR="$i/timepoints"
        FIRST_DIR="$i/firstTimepointMatching"
        HMM_DIR="$i/hmm"
        ORIG="$i/BOLD.nii"
        FIRST="$i/corrected_firstTimepoint.nii.gz"
        HMM="$i/corrected_hmm.nii.gz"
        # remove the existing files (original only)
        rm "$ORIG_DIR"/*
        # unzip the files (original only)
        python utils/splitTimepoints.py -i $ORIG -d $ORIG_DIR
        # Might want to rezip the FIRST and HMM images, but waiting to see what errors pop out
    fi
done

# Calculate correlation ratio matrices
bash calculateCorrelationMatrices-timepoints.sh $BASE #&
bash calculateCorrelationMatrices-first.sh $BASE #&
bash calculateCorrelationMatrices-hmm.sh $BASE #&
#bash calculateCorrelationMatrices-stacking.sh

# Calculate Power et al. displacement and RMS intensity change
bash calculatePowerMetrics.sh $BASE

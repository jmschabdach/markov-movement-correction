#!/bin/bash

BASE='/home/jenna/Research/CHP-PIRC/markov-movement-correction/Controls'
 
# Only use once to split the data 
for i in "$BASE"/* ; do
    if [ -d "$i" ] ; then
        # rm "$i/timepoints/*"
        # rm "$i/firstTimepointMatching/*"
        # rm "$i/hmm/*"
        python utils/splitTimepoints.py -i "$i/BOLD.nii" -d "$i/timepoints/"
        python utils/splitTimepoints.py -i "$i/corrected_firstTimepoint.nii.gz" -d "$i/firstTimepointMatching/"
        python utils/splitTimepoints.py -i "$i/corrected_hmm.nii.gz" -d "$i/hmm/"
    fi
done


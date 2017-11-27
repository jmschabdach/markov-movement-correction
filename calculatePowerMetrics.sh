#!/bin/bash

# Calculate the displacement and voxel intensity change values for all images in the specified directory
# Useage:
# bash calculatePowerMetrics.sh

#BASE='/home/jenna/Research/CHP-PIRC/markov-movement-correction/Controls'
BASE=$1

for i in "$BASE"/* ; do
    if [ -d "$i" ] ; then
        ORIG="$i/BOLD.nii"
        HMM="$i/corrected_hmm.nii.gz"
        FIRST="$i/corrected_firstTimepoint.nii.gz"
        # STACK="$i/corrected_stacking_hmm.nii.gz"
        python utils/stackNiftis.py -d "$i/firstTimepointMatching/" -o $FIRST -c "$i/timepoints/000.nii.gz"
        bash utils/powerMetrics.sh $ORIG &
        bash utils/powerMetrics.sh $FIRST & 
        bash utils/powerMetrics.sh $HMM & 
        # bash utils/powerMetrics.sh $STACK
    fi
done

# check the results

for i in "$BASE"/* ; do
    if [ -d "$i" ] ; then
        # echo $(basename $i) "Num intensity metrics: " $(ls "$i/metrics/" | grep "intensity-metrics" | wc -l ) "Num displacement metrics: " $(ls "$i/metrics/" | grep "displacement-metrics" | wc -l )
        echo $(basename $i) "Num intensity metrics: " $(cat "$i/metrics/corrected_firstTimepoint-intensity-metrics.csv" | wc -l ) "Num displacement metrics: " $(cat "$i/metrics/corrected_firstTimepoint-displacement-metrics.csv" | wc -l )
    fi
done

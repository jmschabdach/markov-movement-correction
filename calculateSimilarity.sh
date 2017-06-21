#!/bin/bash

# use the similarity.sh script to calculate the similarity 
# metric for 2 images

#TEMPLATE='/home/jenna/Research/CHP-PIRC/markov-movement-correction/tmp/timepoints/0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001_000.nii.gz'
# TEMPLATE='/home/pirc/Desktop/Jenna_dev/markov-movement-correction/tmp/timepoints/000.nii.gz'
# TEMPLATE='/home/pirc/processing/FETAL_Axial_BOLD_Motion_Processing/markov-movement-correction/0003_MR2_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001/timepoints/075.nii.gz'
# BASE='/home/pirc/Desktop/Jenna_dev/markov-movement-correction/tmp'
BASE=$1
TEMPLATE=$2

DIR="$BASE/timepoints/*"
# for all the images in the non-registered version
count=0
echo "Timepoint, Similarity, Mutual_Information" > "$BASE/similarities_preregistration.csv"
for img in $DIR ; do
    # compare each image to the template
    sims=$(./utils/similarity.sh $TEMPLATE $img)
    echo $count, $sims >> "$BASE/similarities_preregistration.csv"
    count=$((count+1))
done

DIR="$BASE/markov/*"
# for all the images in the non-registered version
count=0
echo "Timepoint, Similarity, Mutual_Information" > "$BASE/similarities_markov.csv"
for img in $DIR ; do
    # compare each image to the template
    sims=$(./utils/similarity.sh $TEMPLATE $img)
    echo $count, $sims >> "$BASE/similarities_markov.csv"
    count=$((count+1))
done

DIR="$BASE/bifur-markov/*"
# for all the images in the non-registered version
count=0
echo "Timepoint, Similarity, Mutual_Information" > "$BASE/similarities_bifur_markov.csv"
for img in $DIR ; do
    # compare each image to the template
    sims=$(./utils/similarity.sh $TEMPLATE $img)
    echo $count, $sims >> "$BASE/similarities_bifur_markov.csv"
    count=$((count+1))
done

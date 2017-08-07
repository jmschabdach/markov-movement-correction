#!/bin/bash

# use the similarity.sh script to calculate the similarity 
# metric for 2 images

BASE='/home/jms565/Research/CHP-PIRC/markov-movement-correction/0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001_000'
TEMPLATE="$BASE/timepoints/000.nii.gz"

echo $BASE

DIR="$BASE/timepoints/*"
# for all the images in the non-registered version
count=0
echo "Timepoint, Similarity, Mutual_Information" > "$BASE/testing_similarities_preregistration.csv"
for img in $DIR ; do
    # make sure $img != 'template.nii.gz'
    # compare each image to the template
    sims=$(./utils/similarity.sh $TEMPLATE $img)
    echo $count, $sims >> "$BASE/testing_similarities_preregistration.csv"
    count=$((count+1))
done

DIR="$BASE/testing/*"
# for all the images in the non-registered version
count=0
echo "Timepoint, Similarity, Mutual_Information" > "$BASE/testing_similarities_1.csv"
for img in $DIR ; do
    # make sure $img != 'template.nii.gz'
    # compare each image to the template
    sims=$(./utils/similarity.sh $TEMPLATE $img)
    echo $count, $sims >> "$BASE/testing_similarities_1.csv"
    count=$((count+1))
done

DIR="$BASE/testing-08012017/*"
# for all the images in the non-registered version
count=0
echo "Timepoint, Similarity, Mutual_Information" > "$BASE/testing_similarities_0.csv"
for img in $DIR ; do
    # make sure $img != 'template.nii.gz'
    # compare each image to the template
    sims=$(./utils/similarity.sh $TEMPLATE $img)
    echo $count, $sims >> "$BASE/testing_similarities_0.csv"
    count=$((count+1))
done


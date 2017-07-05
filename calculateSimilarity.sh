#!/bin/bash

# use the similarity.sh script to calculate the similarity 
# metric for 2 images

#TEMPLATE='/home/jenna/Research/CHP-PIRC/markov-movement-correction/tmp/timepoints/0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001_000.nii.gz'
# TEMPLATE='/home/pirc/Desktop/Jenna_dev/markov-movement-correction/tmp/timepoints/000.nii.gz'
# TEMPLATE='/home/pirc/processing/FETAL_Axial_BOLD_Motion_Processing/markov-movement-correction/0003_MR2_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001/timepoints/075.nii.gz'
# BASE='/home/pirc/Desktop/Jenna_dev/markov-movement-correction/tmp'
BASE=$1
TEMPLATE=$2

echo $BASE

DIR="$BASE/timepoints/*"
# for all the images in the non-registered version
count=0
echo "Timepoint, Similarity, Mutual_Information" > "$BASE/similarities_preregistration.csv"
for img in $DIR ; do
    # make sure $img != 'template.nii.gz'
    # compare each image to the template
    sims=$(./utils/similarity.sh $TEMPLATE $img)
    echo $count, $sims >> "$BASE/similarities_preregistration.csv"
    count=$((count+1))
done

# DIR="$BASE/hmm/*"
# if [ -d $DIR ] # check that the directory exists
#     # for all the images in the non-registered version
#     count=0
#     echo "Timepoint, Similarity, Mutual_Information" > "$BASE/similarities_hmm.csv"
#     for img in $DIR ; do
#         # compare each image to the template
#         sims=$(./utils/similarity.sh $TEMPLATE $img)
#         echo $count, $sims >> "$BASE/similarities_hmm.csv"
#         count=$((count+1))
#     done
# fi

# DIR="$BASE/bi-hmm/*"
# # for all the images in the bifurcating-markov version
# if [ -d $DIR ] # check that the directory exists
#     count=0
#     echo "Timepoint, Similarity, Mutual_Information" > "$BASE/similarities_bi_hmm.csv"
#     for img in $DIR ; do
#         # compare each image to the template
#         sims=$(./utils/similarity.sh $TEMPLATE $img)
#         echo $count, $sims >> "$BASE/similarities_bi_hmm.csv"
#         count=$((count+1))
#     done
# fi

DIR="$BASE/stacking-hmm/*"
# for all the images in the bifurcating-markov version
# if [ -d $DIR ] ; then # check that the directory exists
count=0
echo "Timepoint, Similarity, Mutual_Information" > "$BASE/similarities_stacking_hmm.csv"
for img in $DIR ; do
    # compare each image to the template
    sims=$(./utils/similarity.sh $TEMPLATE $img)
    echo $count, $sims >> "$BASE/similarities_stacking_hmm.csv"
    count=$((count+1))
done
# fi

DIR="$BASE/stacking-hmm-not-linked/*"
# for all the images in the bifurcating-markov version
# if [ -d $DIR ] ; then # check that the directory exists
count=0
echo "Timepoint, Similarity, Mutual_Information" > "$BASE/similarities_stacking_hmm_not_linked.csv"
for img in $DIR ; do
    # compare each image to the template
    sims=$(./utils/similarity.sh $TEMPLATE $img)
    echo $count, $sims >> "$BASE/similarities_stacking_hmm_not_linked.csv"
    count=$((count+1))
done
# fi

# DIR="$BASE/sequential/*"
# if [ -d $DIR ] # check that the directory exists
#     # for all the images in the non-registered version
#     count=0
#     echo "Timepoint, Similarity, Mutual_Information" > "$BASE/similarities_sequential.csv"
#     for img in $DIR ; do
#         # compare each image to the template
#         sims=$(./utils/similarity.sh $TEMPLATE $img)
#         echo $count, $sims >> "$BASE/similarities_sequential.csv"
#         count=$((count+1))
#     done
# fi

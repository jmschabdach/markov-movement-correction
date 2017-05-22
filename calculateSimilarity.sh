#!/bin/bash

# use the similarity.sh script to calculate the similarity 
# metric for 2 images

#TEMPLATE='/home/jenna/Research/CHP-PIRC/markov-movement-correction/tmp/timepoints/0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001_000.nii.gz'
TEMPLATE='/home/pirc/Desktop/Jenna_dev/markov-movement-correction/tmp/timepoints/000.nii.gz'

DIR='/home/pirc/Desktop/Jenna_dev/markov-movement-correction/tmp/timepoints/*'
# for all the images in the non-registered version
count=0
echo "Timepoint, Similarity, Mutual_Information" > /home/pirc/Desktop/Jenna_dev/markov-movement-correction/similarities_preregistration.csv
for img in $DIR ; do
    # compare each image to the template
    sims=$(./utils/similarity.sh $TEMPLATE $img)
    echo $count, $sims >> /home/pirc/Desktop/Jenna_dev/markov-movement-correction/similarities_preregistration.csv
    count=$((count+1))
done

DIR='/home/pirc/Desktop/Jenna_dev/markov-movement-correction/tmp/noAffine/*'
# for all the images in the registered (non-affine) version
count=0
echo "Timepoint, Similarity, Mutual_Information" > /home/pirc/Desktop/Jenna_dev/markov-movement-correction/similarities_registered_nonAffine.csv
for img in $DIR ; do
     # compare each image to the template
    sims=$(./utils/similarity.sh $TEMPLATE $img)
    echo $count, $sims >> /home/pirc/Desktop/Jenna_dev/markov-movement-correction/similarities_registered_nonAffine.csv
    count=$((count+1))
done

DIR='/home/pirc/Desktop/Jenna_dev/markov-movement-correction/tmp/markov/*'
# for all the images in the non-registered version
count=0
echo "Timepoint, Similarity, Mutual_Information" > /home/pirc/Desktop/Jenna_dev/markov-movement-correction/similarities_markov_nonAffine.csv
for img in $DIR ; do
    # compare each image to the template
    sims=$(./utils/similarity.sh $TEMPLATE $img)
    echo $count, $sims >> /home/pirc/Desktop/Jenna_dev/markov-movement-correction/similarities_markov_nonAffine.csv
    count=$((count+1))
done

DIR='/home/pirc/Desktop/Jenna_dev/markov-movement-correction/tmp/registered/*'
# for all the iamges in the registered (yes affine) version
for img in $DIR ; do
    echo $img
done

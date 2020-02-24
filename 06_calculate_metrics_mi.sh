#!/usr/bin/bash

# Useage:
# bash 06_calculate_metrics.sh /path/to/dir filename.nii.gz

#COHORTDIR=$1
site=$1
TYPE=$2


#for site in $COHORTDIR/*/ ; do # iterate through the sites
#    if [[ $site != *"corrupt"* ]] ; then # if the site directory does not contain the word corrupt
for subject in $site/*/ ; do # iterate through the subjects in the directory
    if [[ -d $subject/$TYPE/ ]] ; then
        echo "Starting calculating MI metrics for $subject"
	MI_OUT="$subject/metrics/$TYPE-fsl-mi-matrix.csv"

	# Calculate mutual information matrix using fsl
	bash /home/jms565/Research/CHP-PIRC/markov-movement-correction/calculate_metrics/calculateMIMatrix.sh $subject/$TYPE/ $MI_OUT
	echo "Finished computing MI matrix using FSL"
    fi
done 
#    fi
#done

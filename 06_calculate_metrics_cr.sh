#!/usr/bin/bash

# Useage:
# bash 06_calculate_metrics.sh /path/to/dir filename.nii.gz

#COHORTDIR=$1
site=$1
TYPE=$2

echo $site
echo $TYPE

#for site in $COHORTDIR/*/ ; do # iterate through the sites
#    if [[ $site != *"corrupt"* ]] ; then # if the site directory does not contain the word corrupt
for subject in $site/*/ ; do # iterate through the subjects in the directory
    echo $subject
    if [[ -d $subject/$TYPE ]] ; then
        echo "Starting calculating metrics for $subject"
	IMGDIR=$subject/$TYPE/
	CORR_OUT="$subject/metrics/$TYPE-correlation-matrix.csv"

        # Calculate correlation ratio matrices
        bash /home/jms565/Research/CHP-PIRC/markov-movement-correction/calculate_metrics/calculateCorrelationMatrix.sh $IMGDIR $CORR_OUT
	echo "Finished computing correlation ratio matrix"
    fi
done 
#    fi
#done

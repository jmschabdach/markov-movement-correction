#!/usr/bin/bash

# Useage:
# bash 06_calculate_metrics.sh /path/to/dir filename.nii.gz

#COHORTDIR=$1
site=$1
FN=$2

#for site in $COHORTDIR/*/ ; do # iterate through the sites
#    if [[ $site != *"corrupt"* ]] ; then # if the site directory does not contain the word corrupt
for subject in $site/*/ ; do # iterate through the subjects in the directory
    if [[ -f $subject/$FN ]] ; then
	IMG=$subject/$FN 

        echo "Starting Power et al. metrics for $IMG"
	OUTDIR=$(dirname "${IMG}")
	TYPE=$(basename "${IMG}" | cut -f 1 -d ".")

	CORR_OUT="$OUTDIR/metrics/$TYPE-correlation-matrix.csv"
	MI_OUT="$OUTDIR/metrics/$TYPE-fsl-mi-matrix.csv"

	# Calculate Power et al. displacement and RMS intensity change
	bash /home/jms565/Research/CHP-PIRC/markov-movement-correction/calculate_metrics/powerMetrics.sh $IMG
	echo "Finished computing Power et al. metrics (FD and DVARS)"
    fi
done 
#    fi
#done

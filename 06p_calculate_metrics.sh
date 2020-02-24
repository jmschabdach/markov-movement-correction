#!/usr/bin/bash

# Useage:
# bash 06_calculate_metrics.sh /path/to/dir filename.nii.gz

#COHORTDIR=$1
site=$1
#FN=$2
FN="corrected_traditional.nii.gz"

count=0

#for site in $COHORTDIR/*/ ; do # iterate through the sites
#    if [[ $site != *"corrupt"* ]] ; then # if the site directory does not contain the word corrupt
for subject in $site/*/ ; do # iterate through the subjects in the directory
#    if [ -f "$subject/BOLD.nii.gz" ] ; then
#        FN="BOLD.nii.gz"
#    fi
#    if [ -f "$subject/BOLD.nii" ] ; then
#        FN="BOLD.nii"
#    fi
    echo $subject/$FN
    bash /home/jms565/Research/CHP-PIRC/markov-movement-correction/calculate_metrics/calculateMetrics.sh $subject/$FN &
    count=$((count+1))

    if [ "$count" -ge 10 ] ; then
        wait
	count=0
    fi
done 
#    fi
#done

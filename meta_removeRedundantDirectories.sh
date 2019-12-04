#!/usr/bin/bash

TYPE=$1
SITE=$2

for subj in $SITE/*/ ; do

    if [[ $TYPE == "original" ]] ; then
        echo "original"
        IMG="$subj/BOLD.nii"
        if [ -f "$IMG.gz" ] ; then
            IMG="$IMG.gz"
        fi

        DIR=""
        if [ -d "$subj/original" ] ; then
            DIR="$subj/original"
        elif [ -d "$subj/timepoints" ] ; then
            DIR="$subj/timepoints"
        fi

    elif [[ $TYPE == "traditional" ]] ; then
        echo "traditional"
        IMG="$subj/corrected_traditional.nii.gz"
        DIR="$subj/traditional"

    elif [[ $TYPE == "dag" ]] ; then
        echo "dag"
        IMG="$subj/corrected_dag.nii.gz"
        DIR="$subj/dag"

    else
        echo "Type not recognized"
        exit -1
    fi

    python meta_compareImgAndDir.py -d $DIR -f $IMG

    # Get the exit status of the script
    status=$?
    
    if [ $status -eq 0 ] ; then
        echo "Redundant directory detected, removing it now"
        #rm -rf $DIR
    else
        echo "OOPS: Sequence and directory contain different info or an error occurred"
    fi
done

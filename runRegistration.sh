#!/bin/bash

# runRegistration.sh
# Usage:
# bash runRegistration.sh <path to site files> <registration method> <registration type>

BASE=$1
METHOD=$2
TYPE=$3

if [ "$TYPE" == "linear" ] ; then
    TYPE="affine"
elif [ "$TYPE" == "affine" ] ; then
    TYPE="affine"
elif [ "$TYPE" == "nonlinear" ] ; then
    TYPE="nonlinear"
else
    echo "The registration type is not valid. Please use 'affine' or 'nonlinear'."
    exit 1
fi

if [ "$METHOD" == "dag" ] ; then
    OUT_FN="corrected_dag.nii.gz" 
elif [ "$METHOD" == "traditional" ] ; then
    OUT_FN="corrected_traditional.nii.gz"
else
    echo "The registration framework method is not valid. Please use 'dag' or 'traditional'."
    exit 1
fi

for i in "$BASE"/* ; do
    if [ -d "$i" ] ; then      
        # Designate the file
        if [ -f "$i/BOLD.nii" ]; then
            SUB_ORIG="$i/BOLD.nii"
        elif [ -f "$i/BOLD.nii.gz" ]; then
            SUB_ORIG="$i/BOLD.nii.gz"
        else
            echo 'The specified file does not exist'
            exit 1
        fi

        if [ -f "$i/$OUT_FN" ] ; then
            # If the registered file already exists for that subject, skip to next subject
            echo "$METHOD $TYPE registration already performed for $i. Skipping to next."
        else
            # Run the registration based on the registration type specified
            bash runAndNotify.sh python globalVolumeRegistration.py -i $SUB_ORIG -o $OUT_FN -t $METHOD -r $TYPE
        fi
    fi
done

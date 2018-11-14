#!/bin/bash

# runRegistration.sh
# Usage:
# bash runRegistration.sh <path to site files> <registration method> <registration type>

BASE=$1
METHOD=$2
TYPE=$3

if [ "$TYPE" == "linear"] ; then
    TYPE="affine"
elif [ "$TYPE" == "affine" ] ; then
    TYPE="affine"
elif [ "$TYPE" == "nonlinear" ] ; then
    TYPE="nonlinear"
else
    echo "The registration type is not valid. Please use 'affine' or 'nonlinear'."
    exit 1

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

        # Run the registration based on the registration type specified
        if [ "$METHOD" == "dag" ] ; then
            bash runAndNotify.sh python globalVolumeRegistration.py -i $SUB_ORIG -o corrected_dag.nii.gz -t dag -r $TYPE
        elif [ "$METHOD" == "traditional" ] ; then
            bash runAndNotify.sh python globalVolumeRegistration.py -i $SUB_ORIG -o corrected_traditional.nii.gz -t traditional -r $TYPE
        else
            echo "The registration type specified is not valid"
            exit 1
        fi
    fi
done

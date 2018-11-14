#!/bin/bash

# runRegistration.sh
# Usage:
# bash runRegistration.sh <path to site files> <registration type>

BASE=$1
TYPE=$2

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
        if [ "$TYPE" == "dag" ] ; then
            bash runAndNotify.sh python globalVolumeRegistration.py -i $SUB_ORIG -o corrected_dag.nii.gz -t dag
        elif [ "$TYPE" == "traditional" ] ; then
            bash runAndNotify.sh python globalVolumeRegistration.py -i $SUB_ORIG -o corrected_traditional.nii.gz -t traditional
        else
            echo "The registration type specified is not valid"
            exit 1
        fi
    fi
done

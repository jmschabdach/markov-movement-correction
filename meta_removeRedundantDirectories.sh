#!/usr/bin/bash

IMG=$1
DIR=$2

python meta_compareImgAndDir.py -d $DIR -f $IMG

# Get the exit status of the script
status=$?

if [ $status -eq 0 ] ; then
    echo "Redundant directory detected"
else
    echo "OOPS: Sequence and directory contain different info"
fi

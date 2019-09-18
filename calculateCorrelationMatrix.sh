#!/bin/bash

# Calculate the correlation ratio matrix for an expanded sequence
# of 3D images. The cross correlation measures the similarity
# between a pair of images, and a smaller cross correlation ratio
# means the images are more similar.
#
# Usage:
# bash calculateCorrelationMatrix.sh [DirectoryOfImages] [OutputFn.csv]

DIR=$1     # assumes there's a / on the end of the path
OUT_FN=$2  # should be a .csv file for the rest of the workflow

# Check to make sure the inputs are what they should be
echo $DIR
echo $OUT_FN

rm $OUT_FN

# Iterate through the collection of images, treat each as a
# template image, and calculate the cross correlation between
# it and all other images
for t in "$DIR"* ; do # grab a new template image
    TEMPLATE=$t
    # Compare the template image to all images
    for f in "$DIR"/* ; do
        # calculate the cross correlation and echo it to a file
        cc=$(./utils/correlation.sh $TEMPLATE $f)
        echo -ne "$cc," >> "$OUT_FN"
    done
    # About to switch to the next template, add a new line in file
    echo -e "" >> "$OUT_FN"
done

# Pass the file to a python script to clean up extra characters on each line
python utils/processCorrelationFile.py -f $OUT_FN

# Notify the user that the correlation ratio matrix has been calculated
echo "Finished calculating correlation ratio matrix."

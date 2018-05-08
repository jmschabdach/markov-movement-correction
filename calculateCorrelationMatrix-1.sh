#!/bin/bash

# Calculate the correlation matrix for an expanded timepoint image.
# Metric is cross correlation
# Usage:
# bash calculateCorrelationMatrix.sh <DirectoryOfImages> <OutputFn.csv>

DIR=$1     # assumes there's a / on the end of the path
OUT_FN=$2  # should be a .csv file for the rest of the workflow

echo $DIR
echo $OUT_FN

rm $OUT_FN

for t in "$DIR"* ; do
    TEMPLATE=$t
    # echo $TEMPLATE
    for f in "$DIR"/* ; do
        # calculate the similarity score and echo it to a file
        cc=$(./utils/correlation-1.sh $TEMPLATE $f)
        echo -ne "$cc," >> "$OUT_FN"
        # echo -ne "$cc, "
    done
    # about to have a new template image, want a new line in file
    echo -e "" >> "$OUT_FN"
done

python utils/processCorrelationFile.py -f $OUT_FN

echo "Finished calculating correlation matrix."

#!/bin/bash

# Calculate all metrics associated with the motion correction analysis
# Updated analysis: 2019-10-12
# Usage: bash calculateMetrics.sh /path/to/image.nii.gz

IMG=$1

OUTDIR=$(dirname "${IMG}")
TYPE=$(basename "${IMG}" | cut -f 1 -d ".")

CORR_OUT="$OUTDIR/metrics/$TYPE-correlation-matrix.csv"
MI_OUT="$OUTDIR/metrics/$TYPE-fsl-mi-matrix.csv"

# Calculate correlation ratio matrices
bash calculateCorrelationMatrix.sh $OUTDIR $CORR_OUT
echo "Finished computing correlation ratio matrix"

# Calculate Power et al. displacement and RMS intensity change
bash powerMetrics.sh $IMG
echo "Finished computing Power et al. metrics (FD and DVARS)"

# Calculate Dice coefficient matrix and mutual information matrix
python calculateDiceMI.py -i $IMG
echo "Finished computing Dice and MI matrices"

# Calculate mutual information matrix using fsl
bash calculateMIMatrix.sh $OUTDIR $MI_OUT
echo "Finished computing MI matrix using FSL"

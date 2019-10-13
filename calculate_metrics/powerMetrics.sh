#!/bin/bash

# Quick script to calculate displacement and voxel intensity 
# changes per Power et al. (2012?)
#
# Usage:
# bash powerMetrics.sh <File> 
#
# Outputs are 2 confound matrices and 2 lists of metrics (all .csv files)

IMG=$1

DIR=$(dirname "${IMG}")
FN=$(basename "${IMG}" | cut -f 1 -d ".")

DISP_CONF="$DIR/metrics/$FN-displacement-confound.csv"
DISP_MET="$DIR/metrics/$FN-displacement-metrics.csv"
INT_CONF="$DIR/metrics/$FN-intensity-confound.csv"
INT_MET="$DIR/metrics/$FN-intensity-metrics.csv"

# want to use the metric --fd for frame displacement 
#     (follows Power et al.; requires motion correction)
# want to use the metric --dvars 
#     (RMS intensity difference of volume N to volume N+1)
# use --nomoco to not have motion correction

# calculate the frame displacement across the entire image
fsl_motion_outliers -i $IMG -o $DISP_CONF --fd -s $DISP_MET

# calculate the RMS intensity difference between volumes
fsl_motion_outliers -i $IMG -o $INT_CONF --nomoco --dvars -s $INT_MET

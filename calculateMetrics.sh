#!/bin/bash

# Calculate all metrics associated with the motion correction analysis
# Updated analysis: 11/18/2017
# Usage: bash calculateMetrics.sh

# Calculate correlation ratio matrices
#bash calculateCorrelationMatrices-timepoints.sh &
#bash calculateCorrelationMatrices-first.sh &
#bash calculateCorrelationMatrices-hmm.sh #&
#bash calculateCorrelationMatrices-stacking.sh

# Calculate Power et al. displacement and RMS intensity change
bash calculatePowerMetrics.sh

#!/bin/bash

# Calculate the correlation matrices for the stacking-hmm test results

# bash calculateCorrelationMatrix.sh 0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001/stacking-hmm/ metrics/correlationMatrix-stackingHmm-5c.csv

BASE='/home/jenna/Research/CHP-PIRC/markov-movement-correction/Controls'

for i in "$BASE"/* ; do
    if [ -d "$i" ] ; then
        mkdir "$i/metrics/"
        HMM="$i/hmm/"
        FIRST="$i/firstTimepointMatching/"
        STACK="$i/stacking-hmm/"
        FIRST_FN="$i/metrics/crossCorrelation-firstTimepoint.csv"
        HMM_FN="$i/metrics/crossCorrelation-hmm.csv"
        STACK_FN="$i/metrics/crossCorrelation-stackingHmm.csv"
        # bash calculateCorrelationMatrix.sh $FIRST $FIRST_FN
        bash calculateCorrelationMatrix-1.sh $HMM $HMM_FN
        # bash calculateCorrelationMatrix.sh $STACK $STACK_FN
    fi
done

#bash calculateCorrelationMatrix-1.sh "/home/jenna/Research/CHP-PIRC/markov-movement-correction/LinearControls/0784_TC_015/hmm/" "/home/jenna/Research/CHP-PIRC/markov-movement-correction/LinearControls/0784_TC_015/metrics/crossCorrelation-hmm.csv"

#bash calculateCorrelationMatrix-1.sh "/home/jenna/Research/CHP-PIRC/markov-movement-correction/LinearControls/0794_TC_026/hmm/" "/home/jenna/Research/CHP-PIRC/markov-movement-correction/LinearControls/0794_TC_026/metrics/crossCorrelation-hmm.csv"

#bash calculateCorrelationMatrix-1.sh "/home/jenna/Research/CHP-PIRC/markov-movement-correction/LinearControls/0799_TC_031/hmm/" "/home/jenna/Research/CHP-PIRC/markov-movement-correction/LinearControls/0799_TC_031/metrics/crossCorrelation-hmm.csv"

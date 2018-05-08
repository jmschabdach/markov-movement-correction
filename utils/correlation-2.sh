#!/bin/bash

#Quick script to calculate correlation ratio
# (similarity metric for registration)
# Usage:
# bash similarity.sh <File1> <File2>
#output is correlation ratio and mutual information

img1=$1
img2=$2

echo "# 1mm scale
setscale 1
setoption smoothing 1
setoption boundguess 1
clear U
setrow UA  1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1
measurecost 7 UA 0 0 0 0 0 0 abs
printparams U" > sched-2.txt

flirt -in $img1 -ref $img2 -schedule sched-2.txt | head -1 | cut -f1 -d' '

rm sched-2.txt

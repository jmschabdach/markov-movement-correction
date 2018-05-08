#!/usr/bin/bash

# Use the image with the highest initial average correlation ratio
# Calculate the pairwise rigid transforms for the frames in that image
python calculateRigidTransforms.py -i /home/jenna/Research/CHP-PIRC/markov-movement-correction/data/NonlinearControls-MI/0533_TC_068_01a/BOLD.nii -d /home/jenna/Research/CHP-PIRC/markov-movement-correction/generateSimulatedData/0533_TC_068_01a

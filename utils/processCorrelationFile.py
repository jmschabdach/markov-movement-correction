"""
Jenna Schabdach 2018

Remove extra characters from the end of the correlation ratio matrix files.

Useage:
    python processCorrelationFile.py -f [correlation ratio matrix file]
"""
from __future__ import print_function
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Process correlation ratio matrix files.")
# Add argument for correlation ratio matrix filename 
parser.add_argument('-f', '--file', type=str, help='Full path to the name of the correlation ratio matrix file to correct')

# now parse the arguments
args = parser.parse_args()

# Get the contents of the file
with open(args.file, 'r') as f:
    # read all lines
    lines = f.readlines()

# Write the contents of the file back to the file, after removing
# extraneous characters from end of line
with open(args.file, 'w') as f:
    # iterate through lines
    for line in lines:
        line = line[:-2]
        # write line
        f.write(line+'\n')

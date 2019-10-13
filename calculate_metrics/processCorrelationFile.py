from __future__ import print_function
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Process correlation matrix files.")
# image filenames
parser.add_argument('-f', '--file', type=str, help='Full path to the name of the file to correct')

# now parse the arguments
args = parser.parse_args()

with open(args.file, 'r') as f:
    # read all lines
    lines = f.readlines()

with open(args.file, 'w') as f:
    # iterate through lines
    for line in lines:
        line = line[:-2]
        # print(line)
        # write line
        f.write(line+'\n')

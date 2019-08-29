"""
calculateRuntimeStats.py

Jenna Schabdach

Given a file of runtimes in the format

Subject | Framework | Runtime
--------+-----------+------------
subjID1 | hmm       | DD:HH:MM:ss
subjID2 | firstTime | DD:HH:MM:ss

Read in the file and calculate the statistics for the runtimes of
each framework.

Statistics calculated: mean, median, min, max

Usage: python calculateRuntimeStats.py [inputFn]

"""

from __future__ import print_function
import pandas as pd
import numpy as np

def parseTimeToInt(timeStr):
    """
    Given a string in the format DD:HH:MM:SS, return an integer representing
    the corresponding number of seconds.

    Inputs:
    - timeStr: string representing a time in the format DD:HH:MM:SS

    Returns:
    - totalSecs: integer; the time in seconds
    """
    # split input string on :
    components = timeStr.split(":")
    components = [float(c) for c in components]
    # multiply
    totalSecs = ((components[0]*24+components[1])*60+components[2])*60+components[3]
    return totalSecs

def formatTime(timeFloat):
    """
    Given a float representation of an amount of time in seconds, parse it into
    a string with the format DD:HH:MM:SS

    Inputs:
    - timeFloat: an amount of time in seconds

    Returns:
    - timeStr: an amount of time in the format DD:HH:MM:SS
    """
    # Divide out days
    days = int(np.floor(timeFloat/24.0/60.0/60.0))
    timeFloat = timeFloat%(24.0*60*60)
    # Divide out hours
    hours = int(np.floor(timeFloat/60.0/60.0))
    timeFloat = timeFloat%(60.0*60)
    # Divide out minutes
    minutes = int(np.floor(timeFloat/60.0))
    timeFloat = timeFloat%60.0
    # Concatenate the string
    timeStr = str(days).zfill(2)+":"+str(hours).zfill(2)+":"+str(minutes).zfill(2)+":"+str(round(timeFloat, 2))
    return timeStr


# Set up the argument parser
# Parse the arguments
fn = "/home/jenna/Research/CHP-PIRC/markov-movement-correction/data/NonlinearControls-CC/timeToRun.csv"
# Read in the file
df = pd.read_csv(fn)
# Determine which frameworks are present
dfHeaders = list(df)
methods = df[dfHeaders[1]].unique().tolist()
print("Method, Minimum, Mean, Median, Maximum")
# for each framework
for method in methods:
    methodLine = method+","
    # isolate the rows for that framework
    methodRows = df[(df[dfHeaders[1]] == method)]
    methodTimes = methodRows[dfHeaders[2]].tolist()
    # convert each time from a string to an integer in seconds
    methodTimes = [parseTimeToInt(i) for i in methodTimes]
    # calculate the min value
    methodLine += formatTime(min(methodTimes))+","
    # calculate the mean value
    methodLine += formatTime(np.mean(methodTimes))+","
    # calculate the median value
    methodLine += formatTime(np.median(methodTimes))+","
    # calculate the max value
    methodLine += formatTime(max(methodTimes))
    print(methodLine)

# format the results
# save or print the results as a table
# save as .csv or latex?


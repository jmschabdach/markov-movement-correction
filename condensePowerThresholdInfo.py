from __future__ import print_function
import argparse
import os
import csv
import numpy as np
import pandas

# Want user to enter the directory
parser = argparse.ArgumentParser(description="Determine how many volumes per subject were corrected sufficiently by the methods")

parser.add_argument("-d", "--dir", type=str, help="Full path of the directory where the subjects to evaluate are")

args = parser.parse_args()

# Set up hard coded variables here
subjDirs = args.dir
dispOutFn = subjDirs+"displacementCounts.csv"
intOutFn = subjDirs+"intensityCounts.csv"
print(subjDirs)
print(dispOutFn)
print(intOutFn)

metricsPre = ["BOLD-",
              "corrected_firstTimepoint-",
              "corrected_hmm-"]#,
              #"corrected_stacking_hmm-"]

# Start processing
# Initialize the output files
with open(dispOutFn, 'w') as f:
    f.write("Subject,Volume,dispBOLD,dispFirstVolume,dispHmm\n") #,dispStackingHmm\n")

with open(intOutFn, 'w') as f:
    f.write("Subject,Volume,intBOLD,intFirstVolume,intHmm\n")#,intStackingHmm\n")

# Iterate through the subject directories
for subjDir in os.listdir(subjDirs):
    metricsPath = subjDirs+"/"+subjDir+'/metrics/'
    if os.path.isdir(os.path.join(subjDirs, subjDir+'/metrics/')):

        # get the list of displacement and intensity metrics files
        dispFns = [metricsPath+prefix+"displacement-metrics.csv" for prefix in metricsPre]
        intFns = [metricsPath+prefix+"intensity-metrics.csv" for prefix in metricsPre]

        displacements = []
        intensities = []
        
        # load information from displacement files
        for fn in dispFns:
            with open(fn, 'r') as inFile:
                reader = csv.reader(inFile, delimiter='\n')
                vals = [float(val) for line in list(reader) for val in line]
                displacements.append(vals)


        # look at intensity files
        for fn in intFns:
            with open(fn, 'r') as inFile:
                reader = csv.reader(inFile, delimiter='\n')
                vals = [float(val) for line in list(reader) for val in line]
                intensities.append(vals)

        # iterate through the second dimension of the lists
        with open(dispOutFn, 'a') as outFn:
            for i in xrange(len(displacements[0])):
                line = subjDir+", " + str(i) + ", "
                line += str(displacements[0][i]) +', '
                line += str(displacements[1][i]) +', '
                line += str(displacements[2][i]) +'\n' #', '
                # line += str(displacements[3][i]) + '\n'
                outFn.write(line)

        with open(intOutFn, 'a') as outFn:
            for i in xrange(len(intensities[0])):
                line = subjDir+", " + str(i) + ", "
                line += str(intensities[0][i]) +', '
                line += str(intensities[1][i]) +', '
                line += str(intensities[2][i]) +'\n' #', '
                # line += str(intensities[3][i]) + '\n'
                outFn.write(line)


print("Finished compiling Power threshold information")

# now figure out how many images are recovered by the methods
# read in the data
dispDf = pandas.read_csv(dispOutFn)
intDf = pandas.read_csv(intOutFn)

outFn = subjDirs+"numImagesRecovered.csv"
line = "method,subject,volumes-displacement,volumes-intensity,volumes-both,image-recovered\n"

# combine the dataframes
combDf = pandas.merge(dispDf, intDf, on=list(dispDf)[:2], how='outer')
print(len(combDf), " == 2550")
combHeaders = list(combDf)

with open(outFn, 'w') as f:
    f.write(line)

    # for each method
    for method in xrange(2,5):
        methodText = combHeaders[method]
        # for each subject
        
        for subject in combDf[combHeaders[0]].unique().tolist():
            subjectText = subject
            # get the counts
#            print(combHeaders[method])
#            print(combHeaders[method+3])
            dispCount = len(combDf[(combDf[combHeaders[0]] == subject ) & (combDf[combHeaders[method]] < 0.2 )]) 
            intCount = len(combDf[(combDf[combHeaders[0]] == subject ) & (combDf[combHeaders[method+3]] < 25 )])
            bothCount = len(combDf[(combDf[combHeaders[0]] == subject ) & (combDf[combHeaders[method]] < 0.2) & (combDf[combHeaders[method+3]] < 25 )])
            recovered = "Not Recovered"
            if bothCount >= 0.5*len(combDf[(combDf[combHeaders[0]] == subject)]):
                recovered = "Recovered"

            # format the line
            line = methodText+","+subjectText+","+str(dispCount)+","+str(intCount)+","+str(bothCount)+", "+recovered+"\n"
            f.write(line)
            
# now generate the contingency tables
"""
# read in the data
dispDf = pandas.read_csv(dispOutFn)
intDf = pandas.read_csv(intOutFn)

# get the headings for the data
dispHeaders = list(dispDf)
intHeaders = list(intDf)

outFn = subjDirs+"contingencyTables.csv"
line = "method1-method2-criteria,both,m1,m2,neither\n"
with open(outFn, 'w') as f:
    f.write(line)

    # get the contingency tables for the displacement
    for m1 in xrange(4):
        for m2 in xrange(m1, 4, 1):
            if m1 != m2:
                # get the number of volumes meeting these conditions
                both = len(dispDf[(dispDf[dispHeaders[m1+2]] < 2.0) & (dispDf[dispHeaders[m2+2]] < 2.0)])
                m1Only = len(dispDf[(dispDf[dispHeaders[m1+2]] < 2.0) & (dispDf[dispHeaders[m2+2]] >= 2.0)])
                m2Only = len(dispDf[(dispDf[dispHeaders[m1+2]] >= 2.0) & (dispDf[dispHeaders[m2+2]] < 2.0)])
                neither = len(dispDf[(dispDf[dispHeaders[m1+2]] >= 2.0) & (dispDf[dispHeaders[m2+2]] >= 2.0)])
                comparison = dispHeaders[m1+2]+'-'+dispHeaders[m2+2]+'-displacement,'
                f.write(comparison+str(both)+','+str(m1Only)+','+str(m2Only)+','+str(neither)+'\n')

    # get the contingency tables for the voxel intensity
    for m1 in xrange(4):
        for m2 in xrange(m1, 4, 1):
            if m1 != m2:
                # get the number of volumes meeting these conditions
                both = len(intDf[(intDf[intHeaders[m1+2]] < 25.0) & (intDf[intHeaders[m2+2]] < 25.0)])
                m1Only = len(intDf[(intDf[intHeaders[m1+2]] < 25.0) & (intDf[intHeaders[m2+2]] >= 25.0)])
                m2Only = len(intDf[(intDf[intHeaders[m1+2]] >= 25.0) & (intDf[intHeaders[m2+2]] < 25.0)])
                neither = len(intDf[(intDf[intHeaders[m1+2]] >= 25.0) & (intDf[intHeaders[m2+2]] >= 25.0)])
                comparison = intHeaders[m1+2]+'-'+intHeaders[m2+2]+'-intensity,'
                f.write(comparison+str(both)+','+str(m1Only)+','+str(m2Only)+','+str(neither)+'\n')

    # get the contingency tables for both
    # combine the dataframes first
    combDf = pandas.concat([dispDf, intDf], axis=1)
    combHeaders = list(combDf)

    for m1 in xrange(4):
        for m2 in xrange(m1, 4, 1):
            if m1 != m2:
                # get the number of volumes meeting these conditions
                both = len(combDf[(combDf[combHeaders[m1+2]] < 2.0) & (combDf[combHeaders[m2+2]] < 2.0) & (combDf[combHeaders[m1+8]] < 25.0) & (combDf[combHeaders[m2+8]] < 25.0)])
                m1Only = len(combDf[(combDf[combHeaders[m1+2]] < 2.0) & (combDf[combHeaders[m2+2]] >= 2.0) & (combDf[combHeaders[m1+8]] < 25.0) & (combDf[combHeaders[m2+8]] >= 25.0)])
                m2Only = len(combDf[(combDf[combHeaders[m1+2]] >= 2.0) & (combDf[combHeaders[m2+2]] < 2.0) & (combDf[combHeaders[m1+8]] >= 25.0) & (combDf[combHeaders[m2+8]] < 25.0)])
                neither = len(combDf)-both-m1Only-m2Only

                comparison = combHeaders[m1+2]+'-'+combHeaders[m2+8]+'-both,'
                f.write(comparison+str(both)+','+str(m1Only)+','+str(m2Only)+','+str(neither)+'\n')


# read in the data
dispDf = pandas.read_csv(dispOutFn)
intDf = pandas.read_csv(intOutFn)

# get the headings for the data
outFn = subjDirs+"methodTables.csv"
line = "method,both,displacement,intensity,neither\n"

# combDf = pandas.concat([dispDf, intDf], axis=1)
#     combHeaders = list(combDf)
methodsList = ["BOLD", "First Volume", "HMM"]#, "Parallelized HMM"]
with open(outFn, 'w') as f:
    f.write(line)

    # get the contingency tables for the displacement
    for t1 in xrange(4):
        both = len(combDf[(combDf[combHeaders[t1+2]] < 2.0) &  (combDf[combHeaders[t1+8]] < 25.0)])
        dispOnly = len(combDf[(combDf[combHeaders[t1+2]] < 2.0) & (combDf[combHeaders[t1+8]] >= 25.0)])
        intOnly = len(combDf[(combDf[combHeaders[t1+2]] >= 2.0) & (combDf[combHeaders[t1+8]] < 25.0)])
        neither = len(combDf[(combDf[combHeaders[t1+2]] >= 2.0) &  (combDf[combHeaders[t1+8]] >= 25.0)])

        comparison = methodsList[t1]
        f.write(comparison+","+str(both)+','+str(dispOnly)+','+str(intOnly)+','+str(neither)+'\n')
"""

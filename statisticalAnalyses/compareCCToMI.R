library(rlist)
library(moments)
library(xtable)

# Read in the counts of volumes meeting the FD and DVARS criteria for each registration type from the files
ccNonLinDispFn <- "/home/jenna/Research/CHP-PIRC/markov-movement-correction/data/NonlinearControls-CC/displacementCounts.csv"
ccNonLinIntFn <- "/home/jenna/Research/CHP-PIRC/markov-movement-correction/data/NonlinearControls-CC/intensityCounts.csv"
miNonlinDispFn <- "/home/jenna/Research/CHP-PIRC/markov-movement-correction/data/NonlinearControls-MI/displacementCounts.csv"
miNonlinIntFn <- "/home/jenna/Research/CHP-PIRC/markov-movement-correction/data/NonlinearControls-MI/intensityCounts.csv"

ccNonLinDispData <- read.csv(ccNonLinDispFn)
ccNonLinIntData <- read.csv(ccNonLinIntFn)
miNonlinDispData <- read.csv(miNonlinDispFn)
miNonlinIntData <- read.csv(miNonlinIntFn)


#---------------------DISPLACEMENTS-----------------------------
dispOrig <- c()
dispGlobalNonLinCC <- c()
dispDAGNonLinCC <- c()
dispGlobalNonLinMI <- c()
dispDAGNonLinMI <- c()

# for each subject in the cc-nonlinear file, add that subject's data to the correct list
for (subj in unique(ccNonLinDispData$Subject)){
  subjDf <- subset(ccNonLinDispData, Subject == subj)
  dispOrig <- c(dispOrig, subjDf$dispBOLD)
  dispGlobalNonLinCC <- c(dispGlobalNonLinCC, subjDf$dispFirstVolume)
  dispDAGNonLinCC <- c(dispDAGNonLinCC, subjDf$dispHmm)
  # dispStacks <- c(dispStacks, subjDf$dispStackingHmm) # Commented out because it isn't currently used
}

# for each subject in the mi-nonlinear file, add that subject's data to the correct list
for (subj in unique(miNonlinDispData$Subject)){
  subjDf <- subset(miNonlinDispData, Subject == subj)
  dispGlobalNonLinMI <- c(dispGlobalNonLinMI, subjDf$dispFirstVolume)
  dispDAGNonLinMI <- c(dispDAGNonLinMI, subjDf$dispHmm)
  # don't grab the original infomration because theoretically it's the same as in the linear file
}

# Perform the Kolmogorov-Smirnov tests for FDs
dispPvals <- c()
dispDs <- c()
# CC Global Nonlinear and MI Global Nonlinear
dispPvals <- c(dispPvals, ks.test(dispGlobalNonLinCC, dispGlobalNonLinMI)$p.value)
dispDs <- c(dispDs, ks.test(dispGlobalNonLinCC, dispGlobalNonLinMI)$statistic)
# CC Global Nonlinear and MI DAG Nonlinear
dispPvals <- c(dispPvals, ks.test(dispGlobalNonLinCC, dispDAGNonLinMI)$p.value)
dispDs <- c(dispDs, ks.test(dispGlobalNonLinCC, dispDAGNonLinMI)$statistic)
# CC DAG Nonlinear and MI Global Nonlinear
dispPvals <- c(dispPvals, ks.test(dispDAGNonLinCC, dispGlobalNonLinMI)$p.value)
dispDs <- c(dispDs, ks.test(dispDAGNonLinCC, dispGlobalNonLinMI)$statistic)
# CC DAG Nonlinear and MI DAG Nonlinear
dispPvals <- c(dispPvals, ks.test(dispDAGNonLinCC, dispDAGNonLinMI)$p.value)
dispDs <- c(dispDs, ks.test(dispDAGNonLinCC, dispDAGNonLinMI)$statistic)

dispPvals
dispDs

#---------------------INTENSITIES-----------------------------
intOrig <- c()
intGlobalNonLinCC <- c()
intDAGNonLinCC <- c()
intGlobalNonLinMI <- c()
intDAGNonLinMI <- c()

# for each subject in the cc-nonlinear file, add that subject's data to the correct list
for (subj in unique(ccNonLinIntData$Subject)){
  subjDf <- subset(ccNonLinIntData, Subject == subj)
  intOrig <- c(intOrig, subjDf$intBOLD)
  intGlobalNonLinCC <- c(intGlobalNonLinCC, subjDf$intFirstVolume)
  intDAGNonLinCC <- c(intDAGNonLinCC, subjDf$intHmm)
  # intStacks <- c(intStacks, subjDf$intStackingHmm) # Commented out because it isn't currently used
}

# for each subject in the mi-nonlinear file, add that subject's data to the correct list
for (subj in unique(miNonlinIntData$Subject)){
  subjDf <- subset(miNonlinIntData, Subject == subj)
  intGlobalNonLinMI <- c(intGlobalNonLinMI, subjDf$intFirstVolume)
  intDAGNonLinMI <- c(intDAGNonLinMI, subjDf$intHmm)
  # don't grab the original infomration because theoretically it's the same as in the linear file
}

# Perform the Kolmogorov-Smirnov tests for FDs
intPvals <- c()
intDs <- c()
# CC Global Nonlinear and MI Global Nonlinear
intPvals <- c(intPvals, ks.test(intGlobalNonLinCC, intGlobalNonLinMI)$p.value)
intDs <- c(intDs, ks.test(intGlobalNonLinCC, intGlobalNonLinMI)$statistic)
# CC Global Nonlinear and MI DAG Nonlinear
intPvals <- c(intPvals, ks.test(intGlobalNonLinCC, intDAGNonLinMI)$p.value)
intDs <- c(intDs, ks.test(intGlobalNonLinCC, intDAGNonLinMI)$statistic)
# CC DAG Nonlinear and MI Global Nonlinear
intPvals <- c(intPvals, ks.test(intDAGNonLinCC, intGlobalNonLinMI)$p.value)
intDs <- c(intDs, ks.test(intDAGNonLinCC, intGlobalNonLinMI)$statistic)
# CC DAG Nonlinear and MI DAG Nonlinear
intPvals <- c(intPvals, ks.test(intDAGNonLinCC, intDAGNonLinMI)$p.value)
intDs <- c(intDs, ks.test(intDAGNonLinCC, intDAGNonLinMI)$statistic)

intPvals
intDs

#------------------------ISMRM 2017

ks.test(dispDAGNonLinCC, dispDAGNonLinMI, alternative = "less")
ks.test(intDAGNonLinCC, intDAGNonLinMI, alternative = "less")


#------------------------Figures

setwd('/home/jenna/Research/CHP-PIRC/markov-movement-correction/figures/')

saveFigureToFile <- function(saveFn, hist1, hist2, title, xlabel, ylabel, legend1, legend2) {
  png(filename = saveFn,
      width = 17, height = 12, units="cm", bg = "white", res=600)
  # plot densities
  plot(density(hist1),
       main = title,
       xlab = xlabel,
       ylab = ylabel)
  lines(density(hist2), lty=2)
  # lines(density(dispBOLDs), lty=1, col="blue")
  abline(v=0.2, col='red')
  legend(11.75, .8, 
         c(legend1, legend2, "Threshold"), 
         lty=c(1,2, 1), lwd=c(2.5, 2.5, 2.5),
         col=c("black", "black", "red"))
  dev.off()
  
}

# save the linear displacement histograms
saveFigureToFile("linearFDHistograms.png", dispDAGLin, dispGlobalLin, "Histogram of Framewise Displacement Changes (Affine)",
                 "Displacement (mm)", "Density", "DAG", "Global")
# save the linear intensity histograms
saveFigureToFile("linearDVARSHistograms.png", intDAGLin, intGlobalLin, "Histogram of Framewise DVARS Changes (Affine)",
                 "RMS Voxel Intensity (units)", "Density", "DAG", "Global")
# save the nonlinear displacement histograms
saveFigureToFile("nonlinearFDHistograms.png", dispDAGNonlin, dispGlobalNonlin, "Histogram of Framewise Displacement Changes (Nonlinear)",
                 "Displacement (mm)", "Density", "DAG", "Global")
# save the nonlinear intensity histograms
saveFigureToFile("nonlinearDVARSHistograms.png", intDAGNonlin, intGlobalNonlin, "Histogram of Framewise DVARS Changes (Nonlinear)",
                 "RMS Voxel Intensity (units)", "Density", "DAG", "Global")

#------ Make tables with statistics about each method

dispMat <- matrix(c(mean(dispOrig), median(dispOrig), sd(dispOrig), skewness(dispOrig), kurtosis(dispOrig), 
                    mean(dispGlobalLin), median(dispGlobalLin), sd(dispGlobalLin), skewness(dispGlobalLin), kurtosis(dispGlobalLin), 
                    mean(dispGlobalNonlin), median(dispGlobalNonlin), sd(dispGlobalNonlin), skewness(dispGlobalNonlin), kurtosis(dispGlobalNonlin), 
                    mean(dispDAGLin), median(dispDAGLin), sd(dispDAGLin), skewness(dispDAGLin), kurtosis(dispDAGLin), 
                    mean(dispDAGNonlin), median(dispDAGNonlin), sd(dispDAGNonlin), skewness(dispDAGNonlin), kurtosis(dispDAGNonlin)),
                  ncol=5)

colnames(dispMat) <- c('Original', 'Global Linear', 'Global Nonlinear', 'DAG Linear', 'DAG Nonlinear')
rownames(dispMat) <- c('Mean', 'Median', 'Standard Deviation', 'Skewness', 'Kurtosis')
dispTable <- as.table(dispMat)
dispTable

intMat <- matrix(c(mean(intOrig), median(intOrig), sd(intOrig), skewness(intOrig), kurtosis(intOrig), 
                   mean(intGlobalLin), median(intGlobalLin), sd(intGlobalLin), skewness(intGlobalLin), kurtosis(intGlobalLin), 
                   mean(intGlobalNonlin), median(intGlobalNonlin), sd(intGlobalNonlin), skewness(intGlobalNonlin), kurtosis(intGlobalNonlin), 
                   mean(intDAGLin), median(intDAGLin), sd(intDAGLin), skewness(intDAGLin), kurtosis(intDAGLin), 
                   mean(intDAGNonlin), median(intDAGNonlin), sd(intDAGNonlin), skewness(intDAGNonlin), kurtosis(intDAGNonlin)),
                 ncol=5)

colnames(intMat) <- c('Original', 'Global Linear', 'Global Nonlinear', 'DAG Linear', 'DAG Nonlinear')
rownames(intMat) <- c('Mean', 'Median', 'Standard Deviation', 'Skewness', 'Kurtosis')
intTable <- as.table(intMat)
intTable

# Now save the tables to files
print(xtable(dispTable, type = "latex"), file = "dispStatisticsTable.tex")
print(xtable(intTable, type='latex'), file='intStatisticsTable.tex')

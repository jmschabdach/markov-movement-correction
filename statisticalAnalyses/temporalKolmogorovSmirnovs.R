library(rlist)
library(moments)
library(xtable)

# Read in the counts of volumes meeting the FD and DVARS criteria for each registration type from the files
linDispFn <- "/home/jenna/Research/CHP-PIRC/markov-movement-correction/data/LinearControls/displacementCounts.csv"
linIntFn <- "/home/jenna/Research/CHP-PIRC/markov-movement-correction/data/LinearControls/intensityCounts.csv"
nonlinDispFn <- "/home/jenna/Research/CHP-PIRC/markov-movement-correction/data/Controls/displacementCounts.csv"
nonlinIntFn <- "/home/jenna/Research/CHP-PIRC/markov-movement-correction/data/Controls/intensityCounts.csv"
linDispData <- read.csv(linDispFn)
linIntData <- read.csv(linIntFn)
nonlinDispData <- read.csv(nonlinDispFn)
nonlinIntData <- read.csv(nonlinIntFn)


#---------------------DISPLACEMENTS-----------------------------
dispOrig <- c()
dispGlobalLin <- c()
dispGlobalNonlin <- c()
dispDAGLin <- c()
dispDAGNonlin <- c()

# for each subject in the linear file, add that subject's data to the correct list
for (subj in unique(linDispData$Subject)){
  subjDf <- subset(linDispData, Subject == subj)
  dispOrig <- c(dispOrig, subjDf$dispBOLD)
  dispGlobalLin <- c(dispGlobalLin, subjDf$dispFirstVolume)
  dispDAGLin <- c(dispDAGLin, subjDf$dispHmm)
  # dispStacks <- c(dispStacks, subjDf$dispStackingHmm) # Commented out because it isn't currently used
}
dispGlobalLin == dispDAGLin

# for each subject in the nonlinear file, add that subject's data to the correct list
for (subj in unique(nonlinDispData$Subject)){
  subjDf <- subset(nonlinDispData, Subject == subj)
  dispGlobalNonlin <- c(dispGlobalNonlin, subjDf$dispFirstVolume)
  dispDAGNonlin <- c(dispDAGNonlin, subjDf$dispHmm)
  # don't grab the original infomration because theoretically it's the same as in the linear file
}
dispGlobalNonlin == dispGlobalLin

# Perform the Kolmogorov-Smirnov tests for FDs
dispPvals <- c()
dispDs <- c()
# Orig and Global Linear
dispPvals <- c(dispPvals, ks.test(dispOrig, dispGlobalLin)$p.value)
dispDs <- c(dispDs, ks.test(dispOrig, dispGlobalLin)$statistic)
# Original and DAG Linear
dispPvals <- c(dispPvals, ks.test(dispOrig, dispDAGLin)$p.value)
dispDs <- c(dispDs, ks.test(dispOrig, dispDAGLin)$statistic)
# Original and Global Nonlinear
dispPvals <- c(dispPvals, ks.test(dispOrig, dispGlobalNonlin)$p.value)
dispDs <- c(dispDs, ks.test(dispOrig, dispGlobalNonlin)$statistic)
# Original and DAG Nonlinear
dispPvals <- c(dispPvals, ks.test(dispOrig, dispDAGNonlin)$p.value)
dispDs <- c(dispDs, ks.test(dispOrig, dispDAGNonlin)$statistic)
# Global Linear and Global Nonlinear
dispPvals <- c(dispPvals, ks.test(dispGlobalLin, dispGlobalNonlin)$p.value)
dispDs <- c(dispDs, ks.test(dispGlobalLin, dispGlobalNonlin)$statistic)
# Global Linear and DAG Linear
dispPvals <- c(dispPvals, ks.test(dispGlobalLin, dispDAGLin)$p.value)
dispDs <- c(dispDs, ks.test(dispGlobalLin, dispDAGLin)$statistic)
# Global Linear and DAG Nonlinear
dispPvals <- c(dispPvals, ks.test(dispGlobalLin, dispDAGNonlin)$p.value)
dispDs <- c(dispDs, ks.test(dispGlobalLin, dispDAGNonlin)$statistic)
# Global Nonlinear and DAG Linear
dispPvals <- c(dispPvals, ks.test(dispGlobalNonlin, dispDAGLin)$p.value)
dispDs <- c(dispDs, ks.test(dispGlobalNonlin, dispDAGLin)$statistic)
# Global Nonlinear and DAG Nonlinear
dispPvals <- c(dispPvals, ks.test(dispGlobalNonlin, dispDAGNonlin)$p.value)
dispDs <- c(dispDs, ks.test(dispGlobalNonlin, dispDAGNonlin)$statistic)
# DAG Linear and DAG Nonlinear
dispPvals <- c(dispPvals, ks.test(dispDAGNonlin, dispDAGLin)$p.value)
dispDs <- c(dispDs, ks.test(dispDAGNonlin, dispDAGLin)$statistic)

dispPvals
dispDs

#---------------------INTENSITIES-----------------------------
intOrig <- c()
intGlobalLin <- c()
intGlobalNonlin <- c()
intDAGLin <- c()
intDAGNonlin <- c()

# for each subject in the linear file, add that subject's data to the correct list
for (subj in unique(linIntData$Subject)){
  subjDf <- subset(linIntData, Subject == subj)
  intOrig <- c(intOrig, subjDf$intBOLD)
  intGlobalLin <- c(intGlobalLin, subjDf$intFirstVolume)
  intDAGLin <- c(intDAGLin, subjDf$intHmm)
  # intStacks <- c(intStacks, subjDf$intStackingHmm) # Commented out because it isn't currently used
}

# for each subject in the nonlinear file, add that subject's data to the correct list
for (subj in unique(nonlinIntData$Subject)){
  subjDf <- subset(nonlinIntData, Subject == subj)
  intGlobalNonlin <- c(intGlobalNonlin, subjDf$intFirstVolume)
  intDAGNonlin <- c(intDAGNonlin, subjDf$intHmm)
  # don't grab the original infomration because theoretically it's the same as in the linear file
}

# Perform the Kolmogorov-Smirnov tests for FDs
intPvals <- c()
intDs <- c()
# Orig and Global Linear
intPvals <- c(intPvals, ks.test(intOrig, intGlobalLin)$p.value)
intDs <- c(intDs, ks.test(intOrig, intGlobalLin)$statistic)
# Original and DAG Linear
intPvals <- c(intPvals, ks.test(intOrig, intDAGLin)$p.value)
intDs <- c(intDs, ks.test(intOrig, intDAGLin)$statistic)
# Original and Global Nonlinear
intPvals <- c(intPvals, ks.test(intOrig, intGlobalNonlin)$p.value)
intDs <- c(intDs, ks.test(intOrig, intGlobalNonlin)$statistic)
# Original and DAG Nonlinear
intPvals <- c(intPvals, ks.test(intOrig, intDAGNonlin)$p.value)
intDs <- c(intDs, ks.test(intOrig, intDAGNonlin)$statistic)
# Global Linear and Global Nonlinear
intPvals <- c(intPvals, ks.test(intGlobalLin, intGlobalNonlin)$p.value)
intDs <- c(intDs, ks.test(intGlobalLin, intGlobalNonlin)$statistic)
# Global Linear and DAG Linear
intPvals <- c(intPvals, ks.test(intGlobalLin, intDAGLin)$p.value)
intDs <- c(intDs, ks.test(intGlobalLin, intDAGLin)$statistic)
# Global Linear and DAG Nonlinear
intPvals <- c(intPvals, ks.test(intGlobalLin, intDAGNonlin)$p.value)
intDs <- c(intDs, ks.test(intGlobalLin, intDAGNonlin)$statistic)
# Global Nonlinear and DAG Linear
intPvals <- c(intPvals, ks.test(intGlobalNonlin, intDAGLin)$p.value)
intDs <- c(intDs, ks.test(intGlobalNonlin, intDAGLin)$statistic)
# Global Nonlinear and DAG Nonlinear
intPvals <- c(intPvals, ks.test(intGlobalNonlin, intDAGNonlin)$p.value)
intDs <- c(intDs, ks.test(intGlobalNonlin, intDAGNonlin)$statistic)
# DAG Linear and DAG Nonlinear
intPvals <- c(intPvals, ks.test(intDAGNonlin, intDAGLin)$p.value)
intDs <- c(intDs, ks.test(intDAGNonlin, intDAGLin)$statistic)

intPvals
intDs


#------------------------ISMRM 2017

ks.test(dispFirsts, dispHmms, alternative = "less")
ks.test(intFirsts, intHmms, alternative = "less")

setwd('/home/jenna/Research/CHP-PIRC/markov-movement-correction/figures/')

saveFigureToFile <- function(saveFn, hist1, hist2, title, xlabel, ylabel, thresh, legend1, legend2, legX, legY) {
  png(filename = saveFn,
      width = 17, height = 12, units="cm", bg = "white", res=600)
  # plot densities
  plot(density(hist1),
       main = title,
       xlab = xlabel,
       ylab = ylabel)
  lines(density(hist2), lty=2)
  # lines(density(dispBOLDs), lty=1, col="blue")
  #abline(v=0.2, col='red')
  abline(v=25, col='red')
  legend(legX, legY, 
         c(legend1, legend2, "Threshold"), 
         lty=c(1,2, 1), lwd=c(2.5, 2.5, 2.5),
         col=c("black", "black", "red"))
  dev.off()
  
}

# save the linear displacement histograms
saveFigureToFile("linearFDHistograms.png", dispDAGLin, dispGlobalLin, "Histogram of Framewise Displacement Changes (Affine)",
                 0.2, "Displacement (mm)", "Density", "DAG", "Global", 11.75, .8)
# save the linear intensity histograms
saveFigureToFile("linearDVARSHistograms.png", intDAGLin, intGlobalLin, "Histogram of Framewise DVARS Changes (Affine)",
                 25, "RMS Voxel Intensity (units)", "Density", "DAG", "Global", 510, .00875)
# save the nonlinear displacement histograms
saveFigureToFile("nonlinearFDHistograms.png", dispDAGNonlin, dispGlobalNonlin, "Histogram of Framewise Displacement Changes (Nonlinear)",
                 0.2, "Displacement (mm)", "Density", "DAG", "Global", 6.25, 1)
# save the nonlinear intensity histograms
saveFigureToFile("nonlinearDVARSHistograms.png", intDAGNonlin, intGlobalNonlin, "Histogram of Framewise DVARS Changes (Nonlinear)",
                 25, "RMS Voxel Intensity (units)", "Density", "DAG", "Global", 510, .01)

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

library(rlist)
library(moments)

# read the data
dispFn <- "/home/jenna/Research/CHP-PIRC/markov-movement-correction/LinearControls/displacementCounts.csv"
intFn <- "/home/jenna/Research/CHP-PIRC/markov-movement-correction/LinearControls/intensityCounts.csv"
dispData <- read.csv(dispFn)
intData <- read.csv(intFn)


#---------------------DISPLACEMENTS-----------------------------
dispBOLDs <- c()
dispFirsts <- c()
dispHmms <- c()
dispStacks <- c()

# for each subject, add the data to a list
for (subj in unique(dispData$Subject)){
  subjDf <- subset(dispData, Subject == subj)
  dispBOLDs <- c(dispBOLDs, subjDf$dispBOLD)
  dispFirsts <- c(dispFirsts, subjDf$dispFirstVolume)
  dispHmms <- c(dispHmms, subjDf$dispHmm)
  dispStacks <- c(dispStacks, subjDf$dispStackingHmm)
}

# Perform the Kolmogorov-Smirnov tests
dispPvals <- c()
dispDs <- c()
# BOLD and First
dispPvals <- c(dispPvals, ks.test(dispBOLDs, dispFirsts)$p.value)
dispDs <- c(dispDs, ks.test(dispBOLDs, dispFirsts)$statistic)
# BOLD and Hmm
dispPvals <- c(dispPvals, ks.test(dispBOLDs, dispHmms)$p.value)
dispDs <- c(dispDs, ks.test(dispBOLDs, dispHmms)$statistic)
# BOLD and Stack
dispPvals <- c(dispPvals, ks.test(dispBOLDs, dispStacks)$p.value)
dispDs <- c(dispDs, ks.test(dispBOLDs, dispStacks)$statistic)
# First and Hmm
dispPvals <- c(dispPvals, ks.test(dispFirsts, dispHmms)$p.value)
dispDs <- c(dispDs, ks.test(dispFirsts, dispHmms)$statistic)
# First and Stack
dispPvals <- c(dispPvals, ks.test(dispFirsts, dispStacks)$p.value)
dispDs <- c(dispDs, ks.test(dispFirsts, dispStacks)$statistic)
# Hmm and Stack
dispPvals <- c(dispPvals, ks.test(dispHmms, dispStacks)$p.value)
dispDs <- c(dispDs, ks.test(dispHmms, dispStacks)$statistic)

dispPvals
dispDs

#---------------------INTENSITIES-----------------------------

intBOLDs <- c()
intFirsts <- c()
intHmms <- c()
intStacks <- c()

# for each subject, add the data to a list
for (subj in unique(intData$Subject)){
  subjDf <- subset(intData, Subject == subj)
  intBOLDs <- c(intBOLDs, abs(subjDf$intBOLD))
  intFirsts <- c(intFirsts, abs(subjDf$intFirstVolume))
  intHmms <- c(intHmms, abs(subjDf$intHmm))
  intStacks <- c(intStacks, abs(subjDf$intStackingHmm))
}

# Perform the Kolmogorov-Smirnov tests
intPvals <- c()
intDs <- c()
# BOLD and First
intPvals <- c(intPvals, ks.test(intBOLDs, intFirsts)$p.value)
intDs <- c(intDs, ks.test(intBOLDs, intFirsts)$statistic)
# BOLD and Hmm
intPvals <- c(intPvals, ks.test(intBOLDs, intHmms)$p.value)
intDs <- c(intDs, ks.test(intBOLDs, intHmms)$statistic)
# BOLD and Stack)
intPvals <- c(intPvals, ks.test(intBOLDs, intStacks)$p.value)
intDs <- c(intDs, ks.test(intBOLDs, intStacks)$statistic)
# First and Hmm
intPvals <- c(intPvals, ks.test(intFirsts, intHmms)$p.value)
intDs <- c(intDs, ks.test(intFirsts, intHmms)$statistic)
# First and Stack
intPvals <- c(intPvals, ks.test(intFirsts, intStacks)$p.value)
intDs <- c(intDs, ks.test(intFirsts, intStacks)$statistic)
# Hmm and Stack
intPvals <- c(intPvals, ks.test(intHmms, intStacks)$p.value)
intDs <- c(intDs, ks.test(intHmms, intStacks)$statistic)

intPvals
intDs

#------------------------ISMRM 2017

ks.test(dispFirsts, dispHmms, alternative = "less")
ks.test(intFirsts, intHmms, alternative = "less")

setwd('/home/jenna/Research/CHP-PIRC/markov-motion-correction/figures/')

png(filename = "powerDensityDisplacement",
    width = 17, height = 12, units="cm", bg = "white", res=600)
# plot densities
plot(density(dispHmms),
     main = "Histogram of Inter-Volume Displacement Changes per Method",
     xlab = "Displacement (mm)",
     ylab = "Density")
lines(density(dispFirsts), lty=2)
# lines(density(dispBOLDs), lty=1, col="blue")
abline(v=0.2, col='red')
legend(11.75, .8, 
       c("HMM", "First Volume", "Threshold"), 
       lty=c(1,2, 1), lwd=c(2.5, 2.5, 2.5),
       col=c("black", "black", "red"))
dev.off()


png(filename = "powerDensityVoxel.png",
    width = 17, height = 12, units="cm", bg = "white", res=600)
# plot intensities
plot(density(intHmms),
     main = "Histogram of Inter-Volume RMS Intensity Changes per Method",
     xlab = "RMS Voxel Intensity (units)",
     ylab = "Density")
lines(density(intFirsts), lty=2)
# lines(density(intBOLDs), lty=1, col="blue")
abline(v=25.0, col='red')
legend(510, .00875, 
       c("HMM", "First Volume", "Threshold"), 
       lty=c(1,2,1), lwd=c(2.5, 2.5, 2.5),
       col=c("black", "black", "red"))
dev.off()

mean(dispFirsts)
median(dispFirsts)
sd(dispFirsts)
skewness(dispFirsts)
kurtosis(dispFirsts)

mean(dispHmms)
median(dispHmms)
sd(dispHmms)
skewness(dispHmms)
kurtosis(dispHmms)


mean(intFirsts)
median(intFirsts)
sd(intFirsts)
skewness(intFirsts)
kurtosis(intFirsts)

mean(intHmms)
median(intHmms)
sd(intHmms)
skewness(intHmms)
kurtosis(intHmms)

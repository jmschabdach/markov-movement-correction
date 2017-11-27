library(rlist)

# read the data
dispFn <- "/home/jenna/Research/CHP-PIRC/markov-movement-correction/LinearControls/tableOfDisplacementAverages.csv"
intFn <- "/home/jenna/Research/CHP-PIRC/markov-movement-correction/LinearControls/tableOfIntensityAverages.csv"
dispData <- read.csv(dispFn)
intData <- read.csv(intFn)

# convert the data to a list (from a dataframe)
dispX <- list.parse(dispData)
intX <- list.parse(intData)

# get the group names
groups <- rep(c('orig', 'first', 'hmm', 'stacking'), 17)

# iterate through the rows in the list
dispY <- c()
intY <- c()
for (name in names(dispX)){
  # structure the list
  dispY <- c(dispY, dispX[[name]]$origAvg, dispX[[name]]$firstAvg, dispX[[name]]$hmmAvg, dispX[[name]]$stackAvg)
  intY <- c(intY, intX[[name]]$origAvg, intX[[name]]$firstAvg, intX[[name]]$hmmAvg, intX[[name]]$stackAvg)
}

# ANOVA: displacements
dispData = data.frame(y=dispY, group = factor(groups))
dispFit = lm(dispY ~ groups, dispData)
anova(dispFit)

# ANOVA: intensitities
intData = data.frame(y=intY, group = factor(groups))
intFit = lm(intY ~ groups, intData)
anova(intFit)
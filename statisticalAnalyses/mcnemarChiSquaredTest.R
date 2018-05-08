library(rlist)

# read the data
fn <- "/home/jenna/Research/CHP-PIRC/markov-movement-correction/LinearControls/contingencyTables.csv"
data <- read.csv(fn)

# convert the data to a list (from a dataframe)
x <- list.parse(data)

# iterate through the rows in the list
pvals <- c()
for (name in names(x)){
  # structure the contingency table 
  cont.table <- matrix(c(x[[name]]$both, x[[name]]$m2, x[[name]]$m1, x[[name]]$neither), nrow=2,
                       dimnames = list("Method 1" = c("Recovered By M1", "Not Recovered By M1"), 
                                       "Method 2" = c("Recovered By M2", "Not Recovered By M2")))
  # calculate the mcnemar test
  t1 <- mcnemar.test(cont.table) #, correct = FALSE)
  # add the p-value from the mcnemar test to the table
  x[[name]]$p_value <- t1$p.value
  pvals <- append(pvals, t1$p.value)
}

print(pvals < 0.05)
print(pvals < 0.001)

# make a pretty data frame
df <- do.call(rbind.data.frame, x)

# save data
outFn <- "/home/jenna/Research/CHP-PIRC/markov-movement-correction/LinearControls/pvalues.csv"
write.csv(df, outFn)
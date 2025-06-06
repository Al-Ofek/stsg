## PM
## Date Format - yyyy-mm-dd HH:MM:SS

# Set working directory and load library
setwd("C:/Users/ofeka/OneDrive - Technion/STSG/stsg python working folder/cosmos")
library(CoSMoS)

data <- read.csv("PM_edited.csv")
data[,'date'] <- as.POSIXct(data[,'date'])
setDT(data) ## converts to data.table

quickTSPlot(data$value)

analysis <- analyzeTS(TS = data, season = "month", dist = "burrIII",
                         acsID = "paretoII", lag.max = 5)

reportTS(analysis, 'dist') + theme_light()

reportTS(analysis, 'dist') ## show seasonal distribution fit
reportTS(analysis, 'acs') ## show seasonal ACS fit
reportTS(analysis, 'stat') ## display basic descriptive statistics

## Create a single sim. for sanity checking
simulation <- simulateTS(analysis, from=NULL, to=NULL)
quickTSPlot(simulation$value)

## Batch production of simulations + Saving to file
n_simulations <- 1000

dt <- data.table()

for (i in 0:(n_simulations-1)){
  print(paste0('Sim. number ',i))
  simulation <- simulateTS(analysis, from=NULL, to=NULL)
  column_name <- paste0('values_',i)
  dt[, (column_name):=simulation[[2]]]
}
fwrite(dt, "PM_cosmos.csv")

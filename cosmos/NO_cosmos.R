## NO
## Date Format - yyyy-mm-dd HH:MM:SS

NO <- read.csv("NO.csv")
NO[,'date'] <- as.POSIXct(NO[,'date'])
setDT(NO) ## converts to data.table

quickTSPlot(NO$value)

NO_analysis <- analyzeTS(TS = NO, season = "month", dist = "ggamma",
                         acsID = "weibull", lag.max = 4)

reportTS(NO_analysis, 'dist') + theme_light() ## show seasonal distribution fit
reportTS(NO_analysis, 'acs') + theme_light() ## show seasonal ACS fit
reportTS(NO_analysis, 'stat') ## display basic descriptive statistics

## Create a single sim. for sanity checking
simulation <- simulateTS(NO_analysis, from=NULL, to=NULL)
quickTSPlot(simulation$value)

n_simulations <- 1000

dt <- data.table()

for (i in 0:(n_simulations-1)){
  print(paste0('Sim. number ',i))
  simulation <- simulateTS(NO_analysis, from=NULL, to=NULL)
  column_name <- paste0('values_',i)
  dt[, (column_name):=simulation[[2]]]
}
fwrite(dt, "NO_cosmos.csv")

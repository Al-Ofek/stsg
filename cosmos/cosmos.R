## copy-paste to get the latest version of CoSMoS

if (!require('devtools')) {install.packages('devtools'); library(devtools)} 

install_github('TycheLab/CoSMoS', upgrade = 'never')

library(CoSMoS)

?`CoSMoS-package`
## Date Format - yyyy-mm-dd HH:MM:SS

ud <- read.csv("ud_edited.csv")
ud[,'date'] <- as.POSIXct(ud[,'date'])
setDT(ud) ## converts to data.table

quickTSPlot(ud$value)

ud_analysis <- analyzeTS(TS = ud, season = "month", dist = "ggamma",
             acsID = "weibull", lag.max = 20)

reportTS(ud_analysis, 'dist') + theme_light()

reportTS(ud_analysis, 'dist') ## show seasonal distribution fit
reportTS(ud_analysis, 'acs') ## show seasonal ACS fit
reportTS(ud_analysis, 'stat') ## display basic descriptive statistics

ud_simulated <- simulateTS(ud_analysis, from=NULL, to=NULL)

data("precip")
p <- analyzeTS(TS = precip, season = "month", dist = "ggamma",
               acsID = "weibull", lag.max = 12)

sim <- simulateTS(p)

pm <- read.csv("PM.csv")
setDT(pm)

a2 <- analyzeTS(TS = pm, season = "month", dist = "ggamma",
                acsID = "weibull", lag.max = 20)

pm_simulated <- simulateTS(a2, from = NULL, to = NULL)

## NO
## Date Format - yyyy-mm-dd HH:MM:SS

NO <- read.csv("NO.csv")
NO[,'date'] <- as.POSIXct(NO[,'date'])
setDT(NO) ## converts to data.table

quickTSPlot(NO$value)

NO_analysis <- analyzeTS(TS = NO, season = "month", dist = "ggamma",
                         acsID = "weibull", lag.max = 4)

reportTS(NO_analysis, 'dist') + theme_light()

reportTS(ud_analysis, 'dist') ## show seasonal distribution fit
reportTS(ud_analysis, 'acs') ## show seasonal ACS fit
reportTS(ud_analysis, 'stat') ## display basic descriptive statistics

n_simulations <- 5

dt <- data.table()

for (i in 0:(n_simulations-1)){
  print(paste0('Sim. number ',i))
  simulation <- simulateTS(NO_analysis, from=NULL, to=NULL)
  column_name <- paste0('values_',i)
  dt[, (column_name):=simulation[[2]]]
}
fwrite(dt, "NO_cosmos.csv")

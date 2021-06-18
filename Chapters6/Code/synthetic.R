library("Synth")
library(SCtools)
data("basque")
basque[85:89, 1:4]
dataprep.out <- dataprep(
  foo = basque,
  predictors = c("school.illit", "school.prim", "school.med",
                 "school.high", "school.post.high", "invest"),
  predictors.op = "mean",
  time.predictors.prior = 1964:1969,
  special.predictors = list(
    list("gdpcap", 1960:1969 , "mean"),
    list("sec.agriculture", seq(1961, 1969, 2), "mean"),
    list("sec.energy", seq(1961, 1969, 2), "mean"),
    list("sec.industry", seq(1961, 1969, 2), "mean"),
    list("sec.construction", seq(1961, 1969, 2), "mean"),
    list("sec.services.venta", seq(1961, 1969, 2), "mean"),
    list("sec.services.nonventa", seq(1961, 1969, 2), "mean"),
    list("popdens", 1969, "mean")),
  dependent = "gdpcap",
  unit.variable = "regionno",
  unit.names.variable = "regionname",
  time.variable = "year",
  treatment.identifier = 17,
  controls.identifier = c(2:16, 18),
  time.optimize.ssr = 1960:1969,
  time.plot = 1955:1997)
synth.out <- synth(data.prep.obj = dataprep.out, method = "BFGS")
path.plot(synth.res = synth.out, dataprep.res = dataprep.out,Ylab = "real per-capita GDP (1986 USD, thousand)", Xlab = "year",Ylim = c(0, 12), Legend = c("Basque country","synthetic Basque country"), Legend.position = "bottomright")
gaps.plot(synth.res = synth.out, dataprep.res = dataprep.out,Ylab = "gap in real per-capita GDP (1986 USD, thousand)", Xlab = "year",Ylim = c(-1.5, 1.5), Main = NA)
tdf<-generate.placebos(dataprep.out ,synth.out )
plot.placebos(tdf  =  tdf , discard.extreme  =  TRUE,mspe.limit  =  5 ,xlab  =  NULL,ylab  =  NULL)
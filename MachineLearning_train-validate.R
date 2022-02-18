# This script is used to train 13 machine learning models with targeted validations

rm(list=ls())
setwd("~/Data")

library(tidyverse)
library(mlr)
library(parallelMap)
library(Metrics)

# set configure options for mlr
configureMlr(on.learner.error = "warn")  

# training set
data_select_train <- read_csv("AQS_d2011_ozone_train.csv") %>% select("Conc.8HrMax":"Met.th")

# construct a task
task <- makeRegrTask(id = "Ozone", data = data_select_train, target = "Conc.8HrMax") 

# data splitting for hyperparameter tuning
load("rin_5cv_spatial.RData")

# data splitting for targeted validations
load("rin_10cv_random.RData")
load("rin_10cv_spatial.RData")
load("rin_10cv_temporal.RData")
load("rin_10cv_external.RData")
load("rin_10cv_1313bysite.RData")

# function used to tune hyperparameters with parallel computing
tune_hyper <- function() {
  parallelStartMulticore(cpus = 50, level = "mlr.tuneParams")
  tune_spatial <- tuneParams(learner = lrn, task = task, resampling = rin_5cv_spatial, measures = list(setAggregation(rmse, test.mean)), par.set = ps, control = ctrl)
  parallelStop()
  return(tune_spatial)
}

# function used to validate model with parallel computing
target_valid <- function() {
  parallelStartMulticore(cpus = 50, level = "mlr.resample")
  cv_random <- resample(learner = lrn, task = task, resampling = rin_10cv_random, measures = list(setAggregation(rmse, test.mean)), models = FALSE, keep.pred = TRUE)
  cv_spatial <- resample(learner = lrn, task = task, resampling = rin_10cv_spatial, measures = list(setAggregation(rmse, test.mean)), models = FALSE, keep.pred = TRUE)
  cv_temporal <- resample(learner = lrn, task = task, resampling = rin_10cv_temporal, measures = list(setAggregation(rmse, test.mean)), models = FALSE, keep.pred = TRUE)
  cv_external <- resample(learner = lrn, task = task, resampling = rin_10cv_external, measures = list(setAggregation(rmse, test.mean)), models = FALSE, keep.pred = TRUE)
  cv_1313bysite <- resample(learner = lrn, task = task, resampling = rin_10cv_1313bysite, measures = list(setAggregation(rmse, test.mean)), models = FALSE, keep.pred = TRUE)
  parallelStop()
  return(list(cv_random = cv_random, cv_spatial = cv_spatial, cv_temporal = cv_temporal, cv_external = cv_external, cv_1313bysite = cv_1313bysite))
}

##### linear regression #####

# validation
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.lm", id = "Ozone.lm"), ppc.center = TRUE, ppc.scale = TRUE)  # make a learner
cv_lm <- target_valid()


##### ridge #####

# tune hyperparameter
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.glmnet", id = "Ozone.ridge", alpha = 0), ppc.center = TRUE, ppc.scale = TRUE)  # make a learner
ps <- makeParamSet(  # searching space
  makeNumericParam(id = "lambda", lower = 0.1, upper = 0.9)
)
ctrl <- makeTuneControlGrid(resolution = 200L)  # grid search
tune_ridge <- tune_hyper()

# validation
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.glmnet", id = "Ozone.ridge", par.vals = c(tune_ridge$learner$next.learner$par.vals, lambda = 0.68)), ppc.center = TRUE, ppc.scale = TRUE)  # learner with optimized hyperparameter
cv_ridge <- target_valid()


##### lasso #####

# tune hypperparameter
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.glmnet", id = "Ozone.lasso", alpha = 1), ppc.center = TRUE, ppc.scale = TRUE)  # make a learner
ps <- makeParamSet(  # searching space
  makeNumericParam(id = "lambda", lower = 0.1, upper = 0.6)
)
ctrl <- makeTuneControlGrid(resolution = 100L)  # grid search
tune_lasso <- tune_hyper()

# validation
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.glmnet", id = "Ozone.lasso", par.vals = c(tune_lasso$learner$next.learner$par.vals, lambda = 0.26)), ppc.center = TRUE, ppc.scale = TRUE)  # make a learner
cv_lasso <- target_valid()


##### elasticnet #####

# tune hyperparameter
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.glmnet", id = "Ozone.elasticnet"), ppc.center = TRUE, ppc.scale = TRUE)  # make a learner
ps <- makeParamSet(  # searching space
  makeNumericParam(id = "lambda", lower = 0.1, upper = 0.5), 
  makeDiscreteParam(id = "alpha", values = seq(0.5, 0.95, length.out = 20))
)
ctrl <- makeTuneControlGrid(resolution = 50L)  # grid search
parallelStartMulticore(cpus = 30, level = "mlr.tuneParams", logging = FALSE)
tune_elasticnet <- tune_hyper()

# validation
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.glmnet", id = "Ozone.elasticnet", par.vals = c(tune_elasticnet$learner$next.learner$par.vals, alpha = 0.8, lambda = 0.35)), ppc.center = TRUE, ppc.scale =TRUE)  # learner with optimized hyperparameter
cv_elasticnet <- target_valid()
  

##### pcr #####

# tune hyperparameter
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.pcr", id = "Ozone.pcr"), ppc.center = TRUE, ppc.scale = TRUE)  # make a learner
ps <- makeParamSet(  # searching space
  makeDiscreteParam(id = "ncomp", values = c(1:(ncol(data_select_train)-1)))
)
ctrl <- makeTuneControlGrid()  # grid search
tune_pcr <- tune_hyper()

# validation
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.pcr", id = "Ozone.pcr", par.vals = c(tune_pcr$learner$next.learner$par.vals, tune_pcr$x)), ppc.center = TRUE, ppc.scale = TRUE)  # learner with optimized hyperparameter
cv_pcr <- target_valid()


##### plsr #####

lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.plsr", id = "Ozone.plsr"), ppc.center = TRUE, ppc.scale = TRUE)  # make a learner
ps <- makeParamSet(  # searching space
  makeDiscreteParam(id = "ncomp", values = c(1:(ncol(data_select_train)-1)))
)
ctrl <- makeTuneControlGrid()  # grid search
tune_plsr <- tune_hyper()

# validation
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.plsr", id = "Ozone.plsr", par.vals = c(tune_plsr$learner$next.learner$par.vals, tune_plsr$x)), ppc.center = TRUE, ppc.scale = TRUE)  # learner with optimized hyperparameter
cv_plsr <- target_valid()
  

##### knn #####

# tune hyperparameter
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.kknn", id = "Ozone.knn"), ppc.center = TRUE, ppc.scale = TRUE)  # make a learner
ps <- makeParamSet(  # searching space
  makeDiscreteParam(id = "k", values = seq(50, 300, by = 2)),
  makeDiscreteParam(id = "distance", values = 1:3),
  makeDiscreteParam(id = "kernel", values = c("gaussian", "triangular"))
)
ctrl <- makeTuneControlGrid()  # grid search
tune_knn <- tune_hyper()

# validation
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.kknn", id = "Ozone.knn", par.vals = c(tune_knn$learner$next.learner$par.vals, tune_knn$x)), ppc.center = TRUE, ppc.scale = TRUE)  # learner with optimized hyperparameter
cv_knn <- target_valid()


##### svr #####

# tune hyperparameter
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.ksvm", id = "Ozone.svr", type = "eps-svr", kernel = "rbfdot"), ppc.center = TRUE, ppc.scale = TRUE)  # make a learner
ps <- makeParamSet(  # searching space
  makeDiscreteParam(id = "epsilon", values = seq(0.05, 0.4, by = 0.05)),
  makeDiscreteParam(id = "C", values = seq(1, 3, by = 0.2)),
  makeDiscreteParam(id = "sigma", values = seq(0.0001, 2, 0.5))
)
ctrl <- makeTuneControlGrid()  # grid search
tune_svr <- tune_hyper()

# validation
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.ksvm", id = "Ozone.svr", par.vals = c(tune_svr$learner$next.learner$par.vals, tune_svr$x)), ppc.center = TRUE, ppc.scale = TRUE)  # learner with optimized hyperparameter
cv_svr <- target_valid()


##### bpnn #####

# tune hyperparameter
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.nnet", id = "Ozone.bpnn", maxit = 100), ppc.center = TRUE, ppc.scale = TRUE)  # make a learner
ps <- makeParamSet(  # searching space
  makeDiscreteParam(id = "size", values = 30)
)
ctrl <- makeTuneControlGrid()  # grid search
tune_bpnn <- tune_hyper()

# validation
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.nnet", id = "Ozone.bpnn", par.vals = c(tune_bpnn$x)), ppc.center = TRUE, ppc.scale = TRUE)  # learner with optimized hyperparameter
cv_svr <- target_valid()


##### dnn #####

# tune hyperparameter
h2o.init()
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.h2o.deeplearning", id = "Ozone.dnn"), ppc.center = TRUE, ppc.scale = TRUE)  # make a learner
ps <- makeParamSet(  # searching space
  makeDiscreteParam(id = "activation", values = c("RectifierWithDropout")),
  makeDiscreteParam(id = "hidden", values = list(a = c(120, 120, 120), b = c(240, 240, 240), c = c(120, 120, 120, 120), d = c(240, 240, 240, 240))),  
  makeDiscreteParam(id = "epochs", values = c(8)),
  makeDiscreteParam(id = "input_dropout_ratio", values = c(0.15)),
  makeDiscreteParam(id = "hidden_dropout_ratios", values = list(a = c(0.05, 0.05, 0.05), b = c(0.1, 0.1, 0.1), c = c(0.2, 0.2, 0.2))),
  makeDiscreteParam(id = "l1", values = c(0, 10^(-6), 10^(-5), 10^(-4), 10^(-3), 10^(-2), 10^(-1))),
  makeDiscreteParam(id = "l2", values = c(0, 10^(-6), 10^(-5), 10^(-3), 10^(-1), 1)),
  makeDiscreteParam(id = "rate", values = c(0.005))
)
ctrl <- makeTuneControlGrid()  # grid search
tune_bpnn <- tune_hyper()

# validation
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.h2o.deeplearning", id = "Ozone.dnn"), ppc.center = TRUE, ppc.scale =TRUE)
lrn <- setHyperPars(lrn, activation ="RectifierWithDropout", hidden = c(240, 240, 240), epochs = 8, input_dropout_ratio = 0.15, hidden_dropout_ratios = c(0.1, 0.1, 0.1), l1 = 0, l2 = 0, rate = 0.005)  # learner with optimized hyperparameter
cv_dnn <- target_valid()


##### rt #####

# tune hyperparameter
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.rpart", id = "Ozone.rt", maxdepth = 30), ppc.center = TRUE, ppc.scale = TRUE)  # make a learner
ps <- makeParamSet(  # searching space
  makeDiscreteParam(id = "minsplit", values = c(200, 400, 800, 1000, 3000)),
  makeDiscreteParam(id = "cp", values = c(10^(-6), 10^(-5), 10^(-4), 10^(-3)))
)
ctrl <- makeTuneControlGrid()  # grid search
tune_rt <- tune_hyper()

# validation
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.rpart", id = "Ozone.rt", par.vals = c(tune_rt$learner$next.learner$par.vals, tune_rt$x)), ppc.center = TRUE, ppc.scale = TRUE)  # learner with optimized hyperparameter
cv_rt <- target_valid()


##### rf #####

# tune hyperparameter
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.ranger", id = "Ozone.rf", num.trees = 600, replace = TRUE), ppc.center = TRUE, ppc.scale = TRUE)  # make a learner
ps <- makeParamSet(  # searching space
  makeDiscreteParam(id = "mtry", values = seq(5, 25, by = 1)),
  makeDiscreteParam(id = "min.node.size", values = seq(10, 20, by = 1))
)
ctrl <- makeTuneControlGrid()  # grid search
tune_rf <- tune_hyper()

# validation
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.ranger", id = "Ozone.rf", par.vals = c(tune_rf$learner$next.learner$par.vals, tune_rf$x)), ppc.center = TRUE, ppc.scale = TRUE)  # learner with optimized hyperparameter
cv_rf <- target_valid()


##### xgboost #####

# tune hyperparameter
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.xgboost", id = "Ozone.xgboost", booster = "gbtree", early_stopping_rounds = 10), ppc.center = TRUE, ppc.scale = TRUE)  # make a learner
ps <- makeParamSet(  # searching space
  makeDiscreteParam(id = "nrounds", values = c(100, 500)), 
  makeDiscreteParam(id = "eta", values = c(0.01, 0.04, 0.1)),
  makeDiscreteParam(id = "max_depth", values = seq(2, 10, by = 2)),
  makeDiscreteParam(id = "min_child_weight", values = c(2, 4, 6)),
  makeDiscreteParam(id = "gamma", values = c(0, 0.1)),
  makeDiscreteParam(id = "subsample", values = c(0.5, 0.6, 0.7, 1)),
  makeDiscreteParam(id = "colsample_bytree", values = c(0.6, 0.8, 1)),
  makeDiscreteParam(id = "lambda", values = 1),
  makeDiscreteParam(id = "alpha", values = c(10^(-6), 10^(-5), 10^(-4)))
)
ctrl <- makeTuneControlGrid()  # optimization algorithm
tune_xgboost <- tune_hyper()

# validation
lrn <- makePreprocWrapperCaret(makeLearner(cl = "regr.xgboost", id = "Ozone.xgboost", par.vals = c(tune_xgboost$x)), ppc.center = TRUE, ppc.scale = TRUE)  # learner with optimized hyperparameter
lrn <- setHyperPars(lrn, booster ="gbtree", early_stopping_rounds = 10) 
cv_xgboost <- target_valid()
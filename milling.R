## R script that demonstrates the implementation of a machine learning-based methodology for predictive maintenance
## on NASA's milling data (Goebel and Agogino, 2007).
## Authored by Anas Mifrani (INP Toulouse, France).

library(ggplot2)
library(ggcorrplot)
library(ggpubr)
library(tensorflow)
library(keras)
library(moments)
library(readr)
library(adlift)
library(car)
library(carData)
library(ModelMetrics)
library(neuralnet)
library(rnn)
library(e1071)
library(tsfeatures)
# Loading NASA's "milling data set" 
# milling_data <- read.csv("mill.csv")
#milling_data <- read.table("mill.txt", sep=",")
milling_data <- read_delim(file="mill.txt", delim=",")
str(milling_data)
# The authors state that the 16 experiments were carried out up to a certain wear limit and sometimes beyond it.
# I assume that that limit defines a critical wear level.
# We would like to discover what that limit is.
# For each individual experiment, at which point (VB) was the cutting terminated?
cases_VB_and_time <- milling_data[, c(1, 3, 4)]
final_VB_values <- array(data=rep(0:0, times=16), dim=16)
for (i in 1:16)
{
  final_VB_values[[i]] <- tail(cases_VB_and_time[cases_VB_and_time$case == i, "VB"], 1)
}
# Case 1 is rather strange: flank wear drops from .5 mm to .44 mm between t = 44 mins and t = 48 mins

VB_k <- 0.65
# Let's take a look at experiment n° 10: DOC = 1.5mm, feed = 0.5mm, material = cast iron
case_1 <- ggplot(data=cases_VB_and_time[cases_VB_and_time$case == 1, ], mapping=aes(x=time, y=VB)) + geom_point() + geom_line() #+ geom_hline(yintercept=VB_k, color = "blue")
case_2 <- ggplot(data=cases_VB_and_time[cases_VB_and_time$case == 2, ], mapping=aes(x=time, y=VB)) + geom_point() + geom_line() #+ geom_hline(yintercept=VB_k, color = "blue")
case_3 <- ggplot(data=cases_VB_and_time[cases_VB_and_time$case == 3, ], mapping=aes(x=time, y=VB)) + geom_point() + geom_line()
case_4 <- ggplot(data=cases_VB_and_time[cases_VB_and_time$case == 4, ], mapping=aes(x=time, y=VB)) + geom_point() + geom_line()
case_5 <- ggplot(data=cases_VB_and_time[cases_VB_and_time$case == 5, ], mapping=aes(x=time, y=VB)) + geom_point() + geom_line()
#case_6 <- ggplot(data=cases_VB_and_time[cases_VB_and_time$case == 6, ], mapping=aes(x=time, y=VB)) + geom_point() + geom_line()
case_16 <- ggplot(data=cases_VB_and_time[cases_VB_and_time$case == 16, ], mapping=aes(x=time, y=VB)) + geom_point() + geom_line()
ggarrange(case_1, case_2, case_3, case_4, case_5)
# In principle, cases 1 and 9 should yield the same results (identical experimental parameters, different set of inserts)
# Let's probe this observation
case_9 <- ggplot(data=cases_VB_and_time[cases_VB_and_time$case == 9, ], mapping=aes(x=time, y=VB)) + geom_point() + geom_line() + geom_hline(yintercept=VB_k, color = "blue")
ggarrange(case_1, case_9)

# The same applies to cases 2 and 10
case_10 <- ggplot(data=cases_VB_and_time[cases_VB_and_time$case == 10, ], mapping=aes(x=time, y=VB)) + geom_point() + geom_line() + geom_hline(yintercept=VB_k, color = "blue")
ggarrange(case_2, case_10)

# Case 16
ggarrange(case_16)

# Using LSTMs to predict flank wear at time t based on previous values (t-1, t-2, ..., t-k)
# Let's "model" case 10

# Let's perform a statistical regression on flank wear, VB.


# Model 1 (LR): VB_i = theta0 + theta1*x_i1 + ... + thetap*x_ip for all observations i. p is the number of independent explanatory variables. 
# LR_model <- lm(formula = VB ~ DOC + feed + material + RMS + variance + skewness + kurtosis + p2p + spec_skewness + spec_kurtosis + wavelet_energy, data = )
experiment_2 <- milling_data[milling_data$case == 2, ]
experiment_2 <- as.data.frame(experiment_2)

RMS_smcAC <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
RMS_smcDC <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
RMS_vib_table <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
RMS_vib_spindle <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
RMS_AE_table <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
RMS_AE_spindle <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
var_smcAC <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
var_smcDC <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
var_vib_table <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
var_vib_spindle <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
var_AE_table <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
var_AE_spindle <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
max_smcAC <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
max_smcDC <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
max_vib_table <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
max_vib_spindle <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
max_AE_table <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
max_AE_spindle <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
skewness_smcAC <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
skewness_smcDC <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
skewness_vib_table <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
skewness_vib_spindle <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
skewness_AE_table <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
skewness_AE_spindle <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
kurtosis_smcAC <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
kurtosis_smcDC <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
kurtosis_vib_table <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
kurtosis_vib_spindle <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
kurtosis_AE_table <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))
kurtosis_AE_spindle <- array(data=rep(0:0, times=nrow(experiment_2)), dim=nrow(experiment_2))

# smcAC: 8 -> 9007
#smcDC: 15368 -> 24367
#vib_table: 30728 -> 39727
#vib_spindle: 46088 -> 55087
#AE_table: 61448 -> 70447
#AE_spindle: 76808 -> 85807


for (i in 1:nrow(experiment_2)) {
  for (j in 8:9007) {
    # RMS
    RMS_smcAC[[i]] <- RMS_smcAC[[i]] + experiment_2[i, j]**2
    RMS_smcDC[[i]] <- RMS_smcDC[[i]] + experiment_2[i, j+15360]**2
    RMS_vib_table[[i]] <- RMS_vib_table[[i]] + experiment_2[i, j+30720]**2
    RMS_vib_spindle[[i]] <- RMS_vib_spindle[[i]] + experiment_2[i, j+46080]**2
    RMS_AE_table[[i]] <- RMS_AE_table[[i]] + experiment_2[i, j+61440]**2
    RMS_AE_spindle[[i]] <- RMS_AE_spindle[[i]] + experiment_2[i, j+76800]**2
  }
  RMS_smcAC[[i]] <- sqrt((1/9000)*RMS_smcAC[[i]])
  RMS_smcDC[[i]] <- sqrt((1/9000)*RMS_smcDC[[i]])
  RMS_vib_table[[i]] <- sqrt((1/9000)*RMS_vib_table[[i]])
  RMS_vib_spindle[[i]] <- sqrt((1/9000)*RMS_vib_spindle[[i]])
  RMS_AE_table[[i]] <- sqrt((1/9000)*RMS_AE_table[[i]])
  RMS_AE_spindle[[i]] <- sqrt((1/9000)*RMS_AE_spindle[[i]])
  
  # Variance
  var_smcAC[[i]] <- var(as.column(experiment_2[i, 8:9007]))
  var_smcDC[[i]] <- var(as.column(experiment_2[i, 15368:24367]))
  var_vib_table[[i]] <- var(as.column(experiment_2[i, 30728:39727]))
  var_vib_spindle[[i]] <- var(as.column(experiment_2[i, 46088:55087]))
  var_AE_table[[i]] <- var(as.column(experiment_2[i, 61448:70447]))
  var_AE_spindle[[i]] <- var(as.column(experiment_2[i, 76808:85807]))
  # Maximum
  max_smcAC[[i]] <- max(as.column(experiment_2[i, 8:9007]))
  max_smcDC[[i]] <- max(as.column(experiment_2[i, 15368:24367]))
  max_vib_table[[i]] <- max(as.column(experiment_2[i, 30728:39727]))
  max_vib_spindle[[i]] <- max(as.column(experiment_2[i, 46088:55087]))
  max_AE_table[[i]] <- max(as.column(experiment_2[i, 61448:70447]))
  max_AE_spindle[[i]] <- max(as.column(experiment_2[i, 76808:85807]))
  # Skewness
  skewness_smcAC[[i]] <- skewness(as.column(experiment_2[i, 8:9007]))
  skewness_smcDC[[i]] <- skewness(as.column(experiment_2[i, 15368:24367]))
  skewness_vib_table[[i]] <- skewness(as.column(experiment_2[i, 30728:39727]))
  skewness_vib_spindle[[i]] <- skewness(as.column(experiment_2[i, 46088:55087]))
  skewness_AE_table[[i]] <- skewness(as.column(experiment_2[i, 61448:70447]))
  skewness_AE_spindle[[i]] <- skewness(as.column(experiment_2[i, 76808:85807]))
  # Kurtosis
  kurtosis_smcAC[[i]] <- kurtosis(as.column(experiment_2[i, 8:9007]))
  kurtosis_smcDC[[i]] <- kurtosis(as.column(experiment_2[i, 15368:24367]))
  kurtosis_vib_table[[i]] <- kurtosis(as.column(experiment_2[i, 30728:39727]))
  kurtosis_vib_spindle[[i]] <- kurtosis(as.column(experiment_2[i, 46088:55087]))
  kurtosis_AE_table[[i]] <- kurtosis(as.column(experiment_2[i, 61448:70447]))
  kurtosis_AE_spindle[[i]] <- kurtosis(as.column(experiment_2[i, 76808:85807]))
}

# Training our model requires a set of VB observations together with the corresponding values of the predictor variables
training_data_exp2 <- cbind(experiment_2[, 1:7], RMS_smcAC, RMS_smcDC, RMS_vib_table, RMS_vib_spindle, RMS_AE_table, RMS_AE_spindle, var_smcAC, var_smcDC, var_vib_table, var_vib_spindle, var_AE_table, var_AE_spindle, max_smcAC, max_smcDC, max_vib_table, max_vib_spindle, max_AE_table, max_AE_spindle, skewness_smcAC, skewness_smcDC, skewness_vib_table, skewness_vib_spindle, skewness_AE_table, skewness_AE_spindle, kurtosis_smcAC, kurtosis_smcDC, kurtosis_vib_table, kurtosis_vib_spindle, kurtosis_AE_table, kurtosis_AE_spindle)
training_data_exp2 <- na.omit(training_data_exp2)

# min-max scaling of VB, time and the temporal features. DOC, feed and material are constant throughout a single experiment.They therefore do not account for the differences in VB values.
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

for (i in 3:ncol(training_data_exp2)) {
  training_data_exp2[, i] <- as.column(normalize(training_data_exp2[, i]))
}


# Fitting the aforementioned LR model to data of experiment 2
LR_exp_2_with_TF <- lm(formula = VB ~ time + RMS_smcAC + RMS_smcDC + RMS_vib_table + RMS_vib_spindle + RMS_AE_table + RMS_AE_spindle + var_smcAC + var_smcDC + var_vib_table + var_vib_spindle + var_AE_table + var_AE_spindle + max_smcAC + max_smcDC + max_vib_table + max_vib_spindle + max_AE_table + max_AE_spindle + skewness_smcAC + skewness_smcDC + skewness_vib_table + skewness_vib_spindle + skewness_AE_table + skewness_AE_spindle + kurtosis_smcAC,
                       data = training_data_exp2)

# Summary of the model
summary(LR_exp_2_with_TF)
sigma(LR_exp_2_with_TF)/mean(training_data_exp2$VB)

# It's likely that some of the independent variables we have selected are strongly correlated.
cor(training_data_exp2)

# Predicted values vs. observed values
## First, let's save the model's response in a data frame.
predicted_df <- data.frame(predicted_VB = predict(LR_exp_2_with_TF, training_data_exp2), time=training_data_exp2$time)
## We can now plot both the observed and the predicted VB values against time.
ggplot(data=training_data_exp2, mapping=aes(x=time, y=VB)) + geom_point(color = "blue") + geom_line(color = "red", data = predicted_df, aes(x = time, y = predicted_VB)) # + geom_line() #+ geom_hline(yintercept=VB_k, color = "blue"))

# What about the model that takes time as the single explanatory variable?
LR_exp_2_without_TF <- lm(formula = VB ~ time, data = training_data_exp2)

# Summary of the model
summary(LR_exp_2_without_TF)

# Predicted values vs. observed values
predicted_df_without_TF <- data.frame(predicted_VB = predict(LR_exp_2_without_TF, training_data_exp2), time=training_data_exp2$time)
ggplot(data=training_data_exp2, mapping=aes(x=time, y=VB)) + geom_point(color = "blue") + geom_line(color = "red", data = predicted_df, aes(x = time, y = predicted_VB)) + geom_line(color = "orange", data = predicted_df_without_TF, aes(x = time, y = predicted_VB)) # + geom_line() #+ geom_hline(yintercept=VB_k, color = "blue"))

# The models we've developed so far pertain to the second experiment only. Let's widen the scope of the model so as to cover the entire set of experiments.
experiments_2_10 <- milling_data[milling_data$case %in% c(2, 10), ]
experiments_2_10 <- as.data.frame(experiments_2_10)


RMS_smcAC <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
RMS_smcDC <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
RMS_vib_table <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
RMS_vib_spindle <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
RMS_AE_table <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
RMS_AE_spindle <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
var_smcAC <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
var_smcDC <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
var_vib_table <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
var_vib_spindle <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
var_AE_table <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
var_AE_spindle <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
max_smcAC <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
max_smcDC <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
max_vib_table <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
max_vib_spindle <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
max_AE_table <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
max_AE_spindle <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
skewness_smcAC <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
skewness_smcDC <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
skewness_vib_table <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
skewness_vib_spindle <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
skewness_AE_table <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
skewness_AE_spindle <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
kurtosis_smcAC <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
kurtosis_smcDC <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
kurtosis_vib_table <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
kurtosis_vib_spindle <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
kurtosis_AE_table <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))
kurtosis_AE_spindle <- array(data=rep(0:0, times=nrow(experiments_2_10)), dim=nrow(experiments_2_10))

for (i in 1:nrow(experiments_2_10)) {
  for (j in 8:9007) {
    # RMS
    RMS_smcAC[[i]] <- RMS_smcAC[[i]] + experiments_2_10[i, j]**2
    RMS_smcDC[[i]] <- RMS_smcDC[[i]] + experiments_2_10[i, j+15360]**2
    RMS_vib_table[[i]] <- RMS_vib_table[[i]] + experiments_2_10[i, j+30720]**2
    RMS_vib_spindle[[i]] <- RMS_vib_spindle[[i]] + experiments_2_10[i, j+46080]**2
    RMS_AE_table[[i]] <- RMS_AE_table[[i]] + experiments_2_10[i, j+61440]**2
    RMS_AE_spindle[[i]] <- RMS_AE_spindle[[i]] + experiments_2_10[i, j+76800]**2
  }
  RMS_smcAC[[i]] <- sqrt((1/9000)*RMS_smcAC[[i]])
  RMS_smcDC[[i]] <- sqrt((1/9000)*RMS_smcDC[[i]])
  RMS_vib_table[[i]] <- sqrt((1/9000)*RMS_vib_table[[i]])
  RMS_vib_spindle[[i]] <- sqrt((1/9000)*RMS_vib_spindle[[i]])
  RMS_AE_table[[i]] <- sqrt((1/9000)*RMS_AE_table[[i]])
  RMS_AE_spindle[[i]] <- sqrt((1/9000)*RMS_AE_spindle[[i]])
  
  # Variance
  var_smcAC[[i]] <- var(as.column(experiments_2_10[i, 8:9007]))
  var_smcDC[[i]] <- var(as.column(experiments_2_10[i, 15368:24367]))
  var_vib_table[[i]] <- var(as.column(experiments_2_10[i, 30728:39727]))
  var_vib_spindle[[i]] <- var(as.column(experiments_2_10[i, 46088:55087]))
  var_AE_table[[i]] <- var(as.column(experiments_2_10[i, 61448:70447]))
  var_AE_spindle[[i]] <- var(as.column(experiments_2_10[i, 76808:85807]))
  # Maximum
  max_smcAC[[i]] <- max(as.column(experiments_2_10[i, 8:9007]))
  max_smcDC[[i]] <- max(as.column(experiments_2_10[i, 15368:24367]))
  max_vib_table[[i]] <- max(as.column(experiments_2_10[i, 30728:39727]))
  max_vib_spindle[[i]] <- max(as.column(experiments_2_10[i, 46088:55087]))
  max_AE_table[[i]] <- max(as.column(experiments_2_10[i, 61448:70447]))
  max_AE_spindle[[i]] <- max(as.column(experiments_2_10[i, 76808:85807]))
  # Skewness
  skewness_smcAC[[i]] <- skewness(as.column(experiments_2_10[i, 8:9007]))
  skewness_smcDC[[i]] <- skewness(as.column(experiments_2_10[i, 15368:24367]))
  skewness_vib_table[[i]] <- skewness(as.column(experiments_2_10[i, 30728:39727]))
  skewness_vib_spindle[[i]] <- skewness(as.column(experiments_2_10[i, 46088:55087]))
  skewness_AE_table[[i]] <- skewness(as.column(experiments_2_10[i, 61448:70447]))
  skewness_AE_spindle[[i]] <- skewness(as.column(experiments_2_10[i, 76808:85807]))
  # Kurtosis
  kurtosis_smcAC[[i]] <- kurtosis(as.column(experiments_2_10[i, 8:9007]))
  kurtosis_smcDC[[i]] <- kurtosis(as.column(experiments_2_10[i, 15368:24367]))
  kurtosis_vib_table[[i]] <- kurtosis(as.column(experiments_2_10[i, 30728:39727]))
  kurtosis_vib_spindle[[i]] <- kurtosis(as.column(experiments_2_10[i, 46088:55087]))
  kurtosis_AE_table[[i]] <- kurtosis(as.column(experiments_2_10[i, 61448:70447]))
  kurtosis_AE_spindle[[i]] <- kurtosis(as.column(experiments_2_10[i, 76808:85807]))
}

training_data_exp2_10 <- cbind(experiments_2_10[, 1:7], RMS_smcAC)
training_data_exp2_10 <- cbind(experiments_2_10[, 1:7], RMS_smcAC, RMS_smcDC, RMS_vib_table, RMS_vib_spindle, RMS_AE_table, RMS_AE_spindle, var_smcAC, var_smcDC, var_vib_table, var_vib_spindle, var_AE_table, var_AE_spindle, max_smcAC, max_smcDC, max_vib_table, max_vib_spindle, max_AE_table, max_AE_spindle, skewness_smcAC, skewness_smcDC, skewness_vib_table, skewness_vib_spindle, skewness_AE_table, skewness_AE_spindle, kurtosis_smcAC, kurtosis_smcDC, kurtosis_vib_table, kurtosis_vib_spindle, kurtosis_AE_table, kurtosis_AE_spindle)
training_data_exp2_10 <- na.omit(training_data_exp2_10)

# min-max scaling of VB, time, experimental parameters (except material, because it is identical in both experiments) and the temporal features.
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

for (i in 3:ncol(training_data_exp2_10)) {
  training_data_exp2_10[, i] <- as.column(normalize(training_data_exp2_10[, i]))
}

# What are the possible LR models that can be designed for experiments 2 and 10?
# Model 1: with temporal features, process information and time
# Model 2: with temporal features and process information
# Model 3: with temporal features and time
# Model 4: with temporal features only
# Model 5: with process information and time
# Model 6: with process information only
# Model 7: with time only

# Model 1
LR_exp_2_10_1 <- lm(formula = VB ~ time + DOC + feed + RMS_smcAC + RMS_smcDC + RMS_vib_table + RMS_vib_spindle + RMS_AE_table + RMS_AE_spindle + var_smcAC + var_smcDC + var_vib_table + var_vib_spindle + var_AE_table + var_AE_spindle + max_smcAC + max_smcDC + max_vib_table + max_vib_spindle + max_AE_table + max_AE_spindle + skewness_smcAC + skewness_smcDC + skewness_vib_table + skewness_vib_spindle + skewness_AE_table + skewness_AE_spindle + kurtosis_smcAC,
                       data = training_data_exp2_10)

# Statistical summary of the model
summary(LR_exp_2_10_1)
# Adjusted R-squared: 0.9338, p-value: 1.537e-06, residual standard error: 0.06883 on 11 degrees of freedom

# Model 2
LR_exp_2_10_2 <- lm(formula = VB ~ DOC + feed + RMS_smcAC + RMS_smcDC + RMS_vib_table + RMS_vib_spindle + RMS_AE_table + RMS_AE_spindle + var_smcAC + var_smcDC + var_vib_table + var_vib_spindle + var_AE_table + var_AE_spindle + max_smcAC + max_smcDC + max_vib_table + max_vib_spindle + max_AE_table + max_AE_spindle + skewness_smcAC + skewness_smcDC + skewness_vib_table + skewness_vib_spindle + skewness_AE_table + skewness_AE_spindle + kurtosis_smcAC,
                    data = training_data_exp2_10)

# Statistical summary of the model
summary(LR_exp_2_10_2)
# Adjusted R-squared: 0.9204, p-value: 1.206e-06, residual standard error: 0.07544 on 12 degrees of freedom

# Model 3
LR_exp_2_10_3 <- lm(formula = VB ~ time + RMS_smcAC + RMS_smcDC + RMS_vib_table + RMS_vib_spindle + RMS_AE_table + RMS_AE_spindle + var_smcAC + var_smcDC + var_vib_table + var_vib_spindle + var_AE_table + var_AE_spindle + max_smcAC + max_smcDC + max_vib_table + max_vib_spindle + max_AE_table + max_AE_spindle + skewness_smcAC + skewness_smcDC + skewness_vib_table + skewness_vib_spindle + skewness_AE_table + skewness_AE_spindle + kurtosis_smcAC,
                    data = training_data_exp2_10)

# Statistical summary of the model
summary(LR_exp_2_10_3)
# Adjusted R-squared: 0.9287, p-value: 6.354e-07, residual standard error: 0.07142 on 12 degrees of freedom

# Model 4
LR_exp_2_10_4 <- lm(formula = VB ~ RMS_smcAC + RMS_smcDC + RMS_vib_table + RMS_vib_spindle + RMS_AE_table + RMS_AE_spindle + var_smcAC + var_smcDC + var_vib_table + var_vib_spindle + var_AE_table + var_AE_spindle + max_smcAC + max_smcDC + max_vib_table + max_vib_spindle + max_AE_table + max_AE_spindle + skewness_smcAC + skewness_smcDC + skewness_vib_table + skewness_vib_spindle + skewness_AE_table + skewness_AE_spindle + kurtosis_smcAC,
                    data = training_data_exp2_10)

# Statistical summary of the model
summary(LR_exp_2_10_4)
# Adjusted R-squared: 0.9235, p-value: 2.629e-07, residual standard error: 0.07397 on 13 degrees of freedom

# Model 5
LR_exp_2_10_5 <- lm(formula = VB ~ time + DOC + feed,
                    data = training_data_exp2_10)

# Statistical summary of the model
summary(LR_exp_2_10_5)
# Adjusted R-squared: 0.8774, p-value: 2.959e-10, residual standard error: 0.09365 on 20 degrees of freedom

# Model 6
LR_exp_2_10_6 <- lm(formula = VB ~ DOC + feed,
                    data = training_data_exp2_10)

# Statistical summary of the model
summary(LR_exp_2_10_6)
# Adjusted R-squared: -0.03775, p-value: 0.6596, residual standard error: 0.2725 on 21 degrees of freedom

# Model 7
LR_exp_2_10_7 <- lm(formula = VB ~ time,
                    data = training_data_exp2_10)

# Statistical summary of the model
summary(LR_exp_2_10_7)
# Adjusted R-squared: 0.7787, p-value: 1.554e-08, residual standard error: 0.1258 on 21 degrees of freedom

# Let's test the first five models on each individual experiment
# Predicted values vs. observed values
predicted_df <- data.frame(predicted_VB_model_1 = predict(LR_exp_2_10_1, training_data_exp2_10), predicted_VB_model_2 = predict(LR_exp_2_10_2, training_data_exp2_10), predicted_VB_model_3 = predict(LR_exp_2_10_3, training_data_exp2_10), predicted_VB_model_4 = predict(LR_exp_2_10_4, training_data_exp2_10), predicted_VB_model_5 = predict(LR_exp_2_10_5, training_data_exp2_10), predicted_VB_model_6 = predict(LR_exp_2_10_6, training_data_exp2_10), predicted_VB_model_7 = predict(LR_exp_2_10_7, training_data_exp2_10), time=training_data_exp2_10$time)
plot_exp_2 <- ggplot(data=training_data_exp2_10[training_data_exp2_10$case == 2,], mapping=aes(x=time, y=VB)) + geom_point(color = "blue") + geom_line(color = "red", data = predicted_df, aes(x = time, y = predicted_VB_model_1)) + geom_line(color = "orange", data = predicted_df, aes(x = time, y = predicted_VB_model_2)) + geom_line(color = "yellow", data = predicted_df, aes(x = time, y = predicted_VB_model_3)) + geom_line(color = "green", data = predicted_df, aes(x = time, y = predicted_VB_model_4)) + geom_line(color = "brown", data = predicted_df, aes(x = time, y = predicted_VB_model_5))  # + geom_line() #+ geom_hline(yintercept=VB_k, color = "blue"))
plot_exp_10 <- ggplot(data=training_data_exp2_10[training_data_exp2_10$case == 10,], mapping=aes(x=time, y=VB)) + geom_point(color = "blue") + geom_line(color = "red", data = predicted_df, aes(x = time, y = predicted_VB_model_1)) + geom_line(color = "orange", data = predicted_df, aes(x = time, y = predicted_VB_model_2)) + geom_line(color = "yellow", data = predicted_df, aes(x = time, y = predicted_VB_model_3)) + geom_line(color = "green", data = predicted_df, aes(x = time, y = predicted_VB_model_4)) + geom_line(color = "brown", data = predicted_df, aes(x = time, y = predicted_VB_model_5))  # + geom_line() #+ geom_hline(yintercept=VB_k, color = "blue"))
ggarrange(plot_exp_2, plot_exp_10)
# Look into this later
# predicted_df_exp_10_non_scaled <- data.frame(predicted_VB_model_1 = predict(LR_exp_2_10_1, ), predicted_VB_model_2 = predict(LR_exp_2_10_2, training_data_exp2_10), predicted_VB_model_3 = predict(LR_exp_2_10_3, training_data_exp2_10), predicted_VB_model_4 = predict(LR_exp_2_10_4, training_data_exp2_10), predicted_VB_model_5 = predict(LR_exp_2_10_5, training_data_exp2_10), predicted_VB_model_6 = predict(LR_exp_2_10_6, training_data_exp2_10), predicted_VB_model_7 = predict(LR_exp_2_10_7, training_data_exp2_10), time=training_data_exp2_10$time)
# plot_exp_10_non_scaled <- ggplot(data=milling_data[milling_data$case == 10, 1:7], mapping=aes(x=time, y=VB)) + geom_point(color="blue")
# ggarrange(plot_exp_2, plot_exp_10, plot_exp_10_non_scaled)


# Row 95 (first run of experiment 12) is singular in that its sensor data wasn't downsized to 9000 time steps as in the rest of the experiments.
experiments <- milling_data[-95, ]
experiments <- as.data.frame(experiments)

# We begin by designing and deriving new temporal features from the sensor data.
# For the three kinds of signals (acoustic emission, vibration and motor current), (Traini et al., 2019) and (Cai et al., 2020) rely on the following statistical properties: 
  ## RMS - 
  ## Variance - 
  ## Maximum -
  ## Skewness - 
  ## Kurtosis - 
  ## Peak-to-peak - 
  ## Spectral skewness - 
  ## Spectral kurtosis -
  ## Wavelet energy -

RMS_smcAC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
RMS_smcDC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
RMS_vib_table <- c(data=rep(0:0, times=nrow(experiments)), use.names=FALSE)
RMS_vib_spindle <- c(data=rep(0:0, times=nrow(experiments)), use.names=FALSE)
RMS_AE_table <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
RMS_AE_spindle <- c(data=rep(0:0, times=nrow(experiments)), use.names=FALSE)
var_smcAC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
var_smcDC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
var_vib_table <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
var_vib_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
var_AE_table <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
var_AE_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
max_smcAC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
max_smcDC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
max_vib_table <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
max_vib_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
max_AE_table <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
max_AE_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
skewness_smcAC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
skewness_smcDC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
skewness_vib_table <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
skewness_vib_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
skewness_AE_table <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
skewness_AE_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
kurtosis_smcAC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
kurtosis_smcDC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
kurtosis_vib_table <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
kurtosis_vib_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
kurtosis_AE_table <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
kurtosis_AE_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
p2p_smcAC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
p2p_smcDC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
p2p_vib_table <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
p2p_vib_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
p2p_AE_table <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
p2p_AE_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
sum_smcAC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
sum_smcDC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
sum_vib_table <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
sum_vib_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
sum_AE_table <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
sum_AE_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
min_smcAC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
min_smcDC <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
min_vib_table <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
min_vib_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
min_AE_table <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
min_AE_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=nrow(experiments))
tsfeatures_smcAC <- array(data=rep(0:0, times=nrow(experiments)), dim=c(nrow(experiments), 16))
colnames(tsfeatures_smcAC) <- c("frequency_smcAC", "nperiods_smcAC", "seasonal_period_smcAC", "trend_smcAC", "spike_smcAC", "linearity_smcAC", "curvature_smcAC", "e_acf1_smcAC", "e_acf10_smcAC", "entropy_smcAC", "x_acf1_smcAC", "x_acf10_smcAC", "diff1_acf1_smcAC", "diff1_acf10_smcAC", "diff2_acf1_smcAC", "diff2_acf10_smcAC")
tsfeatures_smcDC <- array(data=rep(0:0, times=nrow(experiments)), dim=c(nrow(experiments), 16))
colnames(tsfeatures_smcDC) <- c("frequency_smcDC", "nperiods_smcDC", "seasonal_period_smcDC", "trend_smcDC", "spike_smcDC", "linearity_smcDC", "curvature_smcDC", "e_acf1_smcDC", "e_acf10_smcDC", "entropy_smcDC", "x_acf1_smcDC", "x_acf10_smcDC", "diff1_acf1_smcDC", "diff1_acf10_smcDC", "diff2_acf1_smcDC", "diff2_acf10_smcDC")
tsfeatures_vib_table <- array(data=rep(0:0, times=nrow(experiments)), dim=c(nrow(experiments), 16))
colnames(tsfeatures_vib_table) <- c("frequency_vib_table", "nperiods_vib_table", "seasonal_period_vib_table", "trend_vib_table", "spike_vib_table", "linearity_vib_table", "curvature_vib_table", "e_acf1_vib_table", "e_acf10_vib_table", "entropy_vib_table", "x_acf1_vib_table", "x_acf10_vib_table", "diff1_acf1_vib_table", "diff1_acf10_vib_table", "diff2_acf1_vib_table", "diff2_acf10_vib_table")
tsfeatures_vib_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=c(nrow(experiments), 16))
colnames(tsfeatures_vib_spindle) <- c("frequency_vib_spindle", "nperiods_vib_spindle", "seasonal_period_vib_spindle", "trend_vib_spindle", "spike_vib_spindle", "linearity_vib_spindle", "curvature_vib_spindle", "e_acf1_vib_spindle", "e_acf10_vib_spindle", "entropy_vib_spindle", "x_acf1_vib_spindle", "x_acf10_vib_spindle", "diff1_acf1_vib_spindle", "diff1_acf10_vib_spindle", "diff2_acf1_vib_spindle", "diff2_acf10_vib_spindle")
tsfeatures_AE_table <- array(data=rep(0:0, times=nrow(experiments)), dim=c(nrow(experiments), 16))
colnames(tsfeatures_AE_table) <- c("frequency_AE_table", "nperiods_AE_table", "seasonal_period_AE_table", "trend_AE_table", "spike_AE_table", "linearity_AE_table", "curvature_AE_table", "e_acf1_AE_table", "e_acf10_AE_table", "entropy_AE_table", "x_acf1_AE_table", "x_acf10_AE_table", "diff1_acf1_AE_table", "diff1_acf10_AE_table", "diff2_acf1_AE_table", "diff2_acf10_AE_table")
tsfeatures_AE_spindle <- array(data=rep(0:0, times=nrow(experiments)), dim=c(nrow(experiments), 16))
colnames(tsfeatures_AE_spindle) <- c("frequency_AE_spindle", "nperiods_AE_spindle", "seasonal_period_AE_spindle", "trend_AE_spindle", "spike_AE_spindle", "linearity_AE_spindle", "curvature_AE_spindle", "e_acf1_AE_spindle", "e_acf10_AE_spindle", "entropy_AE_spindle", "x_acf1_AE_spindle", "x_acf10_AE_spindle", "diff1_acf1_AE_spindle", "diff1_acf10_AE_spindle", "diff2_acf1_AE_spindle", "diff2_acf10_AE_spindle")


for (i in 1:nrow(experiments)) {
  'for (j in 8:9007) {
    # RMS
    RMS_smcAC[[i]] <- RMS_smcAC[[i]] + experiments[i, j]**2
    RMS_smcDC[[i]] <- RMS_smcDC[[i]] + experiments[i, j+15360]**2
    RMS_vib_table[[i]] <- RMS_vib_table[[i]] + experiments[i, j+30720]**2
    RMS_vib_spindle[[i]] <- RMS_vib_spindle[[i]] + experiments[i, j+46080]**2
    RMS_AE_table[[i]] <- RMS_AE_table[[i]] + experiments[i, j+61440]**2
    RMS_AE_spindle[[i]] <- RMS_AE_spindle[[i]] + experiments[i, j+76800]**2
  }
  RMS_smcAC[[i]] <- sqrt((1/9000)*RMS_smcAC[[i]])
  RMS_smcDC[[i]] <- sqrt((1/9000)*RMS_smcDC[[i]])
  RMS_vib_table[[i]] <- sqrt((1/9000)*RMS_vib_table[[i]])
  RMS_vib_spindle[[i]] <- sqrt((1/9000)*RMS_vib_spindle[[i]])
  RMS_AE_table[[i]] <- sqrt((1/9000)*RMS_AE_table[[i]])
  RMS_AE_spindle[[i]] <- sqrt((1/9000)*RMS_AE_spindle[[i]])
  
  # Variance
  var_smcAC[[i]] <- var(as.column(experiments[i, 8:9007]))
  var_smcDC[[i]] <- var(as.column(experiments[i, 15368:24367]))
  var_vib_table[[i]] <- var(as.column(experiments[i, 30728:39727]))
  var_vib_spindle[[i]] <- var(as.column(experiments[i, 46088:55087]))
  var_AE_table[[i]] <- var(as.column(experiments[i, 61448:70447]))
  var_AE_spindle[[i]] <- var(as.column(experiments[i, 76808:85807]))
  # Maximum
  max_smcAC[[i]] <- max(as.column(experiments[i, 8:9007]))
  max_smcDC[[i]] <- max(as.column(experiments[i, 15368:24367]))
  max_vib_table[[i]] <- max(as.column(experiments[i, 30728:39727]))
  max_vib_spindle[[i]] <- max(as.column(experiments[i, 46088:55087]))
  max_AE_table[[i]] <- max(as.column(experiments[i, 61448:70447]))
  max_AE_spindle[[i]] <- max(as.column(experiments[i, 76808:85807]))
  # Skewness
  skewness_smcAC[[i]] <- skewness(as.column(experiments[i, 8:9007]))
  skewness_smcDC[[i]] <- skewness(as.column(experiments[i, 15368:24367]))
  skewness_vib_table[[i]] <- skewness(as.column(experiments[i, 30728:39727]))
  skewness_vib_spindle[[i]] <- skewness(as.column(experiments[i, 46088:55087]))
  skewness_AE_table[[i]] <- skewness(as.column(experiments[i, 61448:70447]))
  skewness_AE_spindle[[i]] <- skewness(as.column(experiments[i, 76808:85807]))
  # Kurtosis
  kurtosis_smcAC[[i]] <- kurtosis(as.column(experiments[i, 8:9007]))
  kurtosis_smcDC[[i]] <- kurtosis(as.column(experiments[i, 15368:24367]))
  kurtosis_vib_table[[i]] <- kurtosis(as.column(experiments[i, 30728:39727]))
  kurtosis_vib_spindle[[i]] <- kurtosis(as.column(experiments[i, 46088:55087]))
  kurtosis_AE_table[[i]] <- kurtosis(as.column(experiments[i, 61448:70447]))
  kurtosis_AE_spindle[[i]] <- kurtosis(as.column(experiments[i, 76808:85807]))
  # Peak-to-peak
  p2p_smcAC[[i]] <- max(experiments[i, 8:9007]) - min(experiments[i, 8:9007])
  p2p_smcDC[[i]] <- max(experiments[i, 15368:24367]) - min(experiments[i, 15368:24367])
  p2p_vib_table[[i]] <- max(experiments[i, 30728:39727]) - min(experiments[i, 30728:39727])
  p2p_vib_spindle[[i]] <- max(experiments[i, 46088:55087]) - min(experiments[i, 46088:55087])
  p2p_AE_table[[i]] <- max(experiments[i, 61448:70447]) - min(experiments[i, 61448:70447])
  p2p_AE_spindle[[i]] <- max(experiments[i, 76808:85807]) - min(experiments[i, 76808:85807])
  # Sum
  sum_smcAC[[i]] <- sum(experiments[i, 8:9007])
  sum_smcDC[[i]] <- sum(experiments[i, 15368:24367])
  sum_vib_table[[i]] <- sum(experiments[i, 30728:39727])
  sum_vib_spindle[[i]] <- sum(experiments[i, 46088:55087])
  sum_AE_table[[i]] <- sum(experiments[i, 61448:70447])
  sum_AE_spindle[[i]] <- sum(experiments[i, 76808:85807])
  # Min
  min_smcAC[[i]] <- min(experiments[i, 8:9007])
  min_smcDC[[i]] <- min(experiments[i, 15368:24367])
  min_vib_table[[i]] <- min(experiments[i, 30728:39727])
  min_vib_spindle[[i]] <- min(experiments[i, 46088:55087])
  min_AE_table[[i]] <- min(experiments[i, 61448:70447])
  min_AE_spindle[[i]] <- min(experiments[i, 76808:85807])'
  # Features provided by tsfeatures
  tsfeatures_smcAC[i, ] <- as.row(tsfeatures(ts(t(experiments[i, 8:9007]))))
  tsfeatures_smcDC[i, ] <- as.row(tsfeatures(ts(t(experiments[i, 15368:24367]))))
  tsfeatures_vib_table[i, ] <- as.row(tsfeatures(ts(t(experiments[i, 30728:39727]))))
  tsfeatures_vib_spindle[i, ] <- as.row(tsfeatures(ts(t(experiments[i, 46088:55087]))))
  tsfeatures_AE_table[i, ] <- as.row(tsfeatures(ts(t(experiments[i, 61448:70447]))))
  tsfeatures_AE_spindle[i, ] <- as.row(tsfeatures(ts(t(experiments[i, 76808:85807]))))
}

# Now that the features have been computed for each of the signals, let's append them to the data set
data_with_temporal_features <- cbind(experiments[, 1:7], RMS_smcAC, RMS_smcDC, RMS_vib_table, RMS_vib_spindle, RMS_AE_table, RMS_AE_spindle, var_smcAC, var_smcDC, var_vib_table, var_vib_spindle, var_AE_table, var_AE_spindle, max_smcAC, max_smcDC, max_vib_table, max_vib_spindle, max_AE_table, max_AE_spindle, skewness_smcAC, skewness_smcDC, skewness_vib_table, skewness_vib_spindle, skewness_AE_table, skewness_AE_spindle, kurtosis_smcAC, kurtosis_smcDC, kurtosis_vib_table, kurtosis_vib_spindle, kurtosis_AE_table, kurtosis_AE_spindle, p2p_smcAC, p2p_smcDC, p2p_vib_table, p2p_vib_spindle, p2p_AE_table, p2p_AE_spindle, sum_smcAC, sum_smcDC, sum_vib_table, sum_vib_spindle, sum_AE_table, sum_AE_spindle, min_smcAC, min_smcDC, min_vib_table, min_vib_spindle, min_AE_table, min_AE_spindle, tsfeatures_smcAC, tsfeatures_smcDC, tsfeatures_vib_table, tsfeatures_vib_spindle, tsfeatures_AE_table, tsfeatures_AE_spindle, row.names = NULL)
# Runs that do not have a corresponding VB value are discarded from the VB modelling phase.
data_with_temporal_features <- na.omit(data_with_temporal_features)

# What correlations are there between the 37 features?
ggcorrplot::ggcorrplot(cor(data_with_temporal_features), hc.order = TRUE)
## Temporal features are often perfectly correlated. For example:
plot_RMS_smcAC <- ggplot(data = data_with_temporal_features[data_with_temporal_features$case == 10, ], mapping = aes(time, RMS_smcAC)) + geom_point(color = "blue")
plot_var_smcAC <- ggplot(data = data_with_temporal_features[data_with_temporal_features$case == 10, ], mapping = aes(time, var_smcAC)) + geom_point(color = "blue")
plot_max_smcAC <- ggplot(data = data_with_temporal_features[data_with_temporal_features$case == 10, ], mapping = aes(time, max_smcAC)) + geom_point(color = "blue")
plot_skewness_smcAC <- ggplot(data = data_with_temporal_features[data_with_temporal_features$case == 10, ], mapping = aes(time, skewness_smcAC)) + geom_point(color = "blue")
plot_kurtosis_smcAC <- ggplot(data = data_with_temporal_features[data_with_temporal_features$case == 10, ], mapping = aes(time, kurtosis_smcAC)) + geom_point(color = "blue")
plot_p2p_smcAC <- ggplot(data = data_with_temporal_features[data_with_temporal_features$case == 10, ], mapping = aes(time, p2p_smcAC)) + geom_point(color = "blue")
plot_VB_case_10 <- ggplot(data = data_with_temporal_features[data_with_temporal_features$case == 10, ], mapping = aes(time, VB)) + geom_point(color = "red")
ggarrange(plot_RMS_smcAC, plot_var_smcAC, plot_max_smcAC, plot_skewness_smcAC, plot_kurtosis_smcAC, plot_p2p_smcAC, plot_VB_case_10)
## Now, what variables have the greatest influence on VB?
cor(data_with_temporal_features[-3], data_with_temporal_features$VB)
## time, run, ...

# Let's perform PCA on the scaled data
pca <- princomp(data_scaled[, 4:ncol(data_scaled)])
plot(cumsum(pca$sdev^2 / sum(pca$sdev^2)), type="b") 
plot(pca$sdev^2)

# DOC, feed and material are categorical variables. Let's code them in 0s and 1s.
data_with_TF_coded <- data_with_temporal_features
data_with_TF_coded$DOC <- ifelse(data_with_TF_coded$DOC == 1.5, 1, 0)
data_with_TF_coded$feed <- ifelse(data_with_TF_coded$feed == 0.5, 1, 0)
data_with_TF_coded$material <- ifelse(data_with_TF_coded$material == 2, 1, 0)

# min-max scaling of the data (except case)
data_scaled <- data_with_TF_coded
for (i in 2:ncol(data_scaled)) {
  data_scaled[, i] <- c(as.column(normalize(data_scaled[, i])))
}

# Let's partition the observations into those intended to train the models and those intended to assess their performance.
## Cases 11, 12, 15 and 16 are test cases.
test_scaled <- data_scaled[data_scaled$case == 12 | data_scaled$case == 15 | data_scaled$case == 11 | data_scaled$case == 16, ]
train_scaled <- data_scaled[!(data_scaled$case == 12  | data_scaled$case == 15 | data_scaled$case == 11 | data_scaled$case == 16), ]

# Model 1: VB = b0 + b1*run + b2*max_AE_spindle + b3*skewness_vib_spindle + b4*material + epsilon
LR_1 <- lm(formula = VB ~ run + max_AE_spindle + skewness_vib_spindle + material,
                     data = train_scaled)
summary(LR_1)

LR_1_bis <- lm(formula = VB ~ run + max_AE_spindle + skewness_vib_spindle + 
                 material + diff1_acf10_AE_spindle + trend_AE_table, data = train_scaled)
summary(LR_1_bis)

## Let's plot predicted vs. actual VB values for cases 11, 12, 15 and 16 (with 95% prediction intervals)
real_time <- test_scaled$time * (max(data_with_TF_coded$time) - min(data_with_TF_coded$time)) + min(data_with_TF_coded$time)
real_run <- test_scaled$run * (max(data_with_TF_coded$run) - min(data_with_TF_coded$run)) + min(data_with_TF_coded$run)
real_predictions <- predict(LR_1, test_scaled, interval = "predict") * (max(data_with_TF_coded$VB) - min(data_with_TF_coded$VB)) + min(data_with_TF_coded$VB)
real_VB <- test_scaled$VB * (max(data_with_TF_coded$VB) - min(data_with_TF_coded$VB)) + min(data_with_TF_coded$VB)
LR_1_results <- data.frame(run = real_run, time = real_time, case = test_scaled$case, actual = real_VB, predicted = real_predictions[, 1], lbound = real_predictions[, 2], ubound = real_predictions[, 3])
plot(LR_1_results$actual, LR_1_results$predicted, col = "red", 
     main = 'VB observées vs. VB prédites')
abline(0, 1, lwd = 2)
ggplot(data=LR_1_results_train[LR_1_results_train$case == 1, ], mapping=aes(x=time, y=actual)) + geom_point(color = "blue") + geom_line(color = "red", data = LR_1_results_train[LR_1_results_train$case == 1, ], aes(x = time, y = predicted)) + geom_hline(yintercept = VB_k)

real_predictions_bis <- predict(LR_1_bis, test_scaled, interval = "predict") * (max(data_with_TF_coded$VB) - min(data_with_TF_coded$VB)) + min(data_with_TF_coded$VB)
LR_1_bis_results <- data.frame(run = real_run, time = real_time, case = test_scaled$case, actual = real_VB, predicted = real_predictions_bis[, 1], lbound = real_predictions_bis[, 2], ubound = real_predictions_bis[, 3])
plot(LR_1_bis_results$actual, LR_1_bis_results$predicted, col = "red", 
     main = 'VB observées vs. VB prédites')
abline(0, 1, lwd = 2)
RMSE_LR_1_bis_test <- rmse(actual = LR_1_bis_results$actual, predicted = LR_1_bis_results$predicted)


## Let's compute model 1's RMSE, MAE and average accuracy in cases 11, 12, 15 and 16
RMSE_LR_1_exp_11 <- rmse(actual = LR_1_results[LR_1_results$case == 11, "actual"], predicted = LR_1_results[LR_1_results$case == 11, "predicted"])
RMSE_LR_1_exp_12 <- rmse(actual = LR_1_results[LR_1_results$case == 12, "actual"], predicted = LR_1_results[LR_1_results$case == 12, ]$predicted)
RMSE_LR_1_exp_15 <- rmse(actual = LR_1_results[LR_1_results$case == 15, "actual"], predicted = LR_1_results[LR_1_results$case == 15, ]$predicted)
RMSE_LR_1_exp_16 <- rmse(actual = LR_1_results[LR_1_results$case == 16, "actual"], predicted = LR_1_results[LR_1_results$case == 16, ]$predicted)
RMSE_LR_1_test <- rmse(actual = LR_1_results$actual, predicted = LR_1_results$predicted)
MAE_LR_1_exp_11 <- mae(actual = LR_1_results[LR_1_results$case == 11, "actual"], predicted = LR_1_results[LR_1_results$case == 11, ]$predicted)
MAE_LR_1_exp_12 <- mae(actual = LR_1_results[LR_1_results$case == 12, "actual"], predicted = LR_1_results[LR_1_results$case == 12, ]$predicted)
MAE_LR_1_exp_15 <- mae(actual = LR_1_results[LR_1_results$case == 15, "actual"], predicted = LR_1_results[LR_1_results$case == 15, ]$predicted)
MAE_LR_1_exp_16 <- mae(actual = LR_1_results[LR_1_results$case == 16, "actual"], predicted = LR_1_results[LR_1_results$case == 16, ]$predicted)
MAE_LR_1_test <- mae(actual = LR_1_results$actual, predicted = LR_1_results$predicted)
avg_accuracy_LR_1_exp_11 <- MAE_LR_1_exp_11 / mean(LR_1_results[LR_1_results$case == 11, "actual"])
avg_accuracy_LR_1_exp_12 <- MAE_LR_1_exp_12 / mean(LR_1_results[LR_1_results$case == 12, "actual"])
avg_accuracy_LR_1_exp_15 <- MAE_LR_1_exp_15 / mean(LR_1_results[LR_1_results$case == 15, "actual"])
avg_accuracy_LR_1_exp_16 <- MAE_LR_1_exp_16 / mean(LR_1_results[LR_1_results$case == 16, "actual"])
avg_accuracy_LR_1_test <- MAE_LR_1_test / mean(LR_1_results$actual)

## Let's compare the actual to the predicted increase in flank wear in cases 11, 12, 15 and 16
plot_exp_15 <- ggplot(data=LR_1_results[LR_1_results$case == 15,], mapping=aes(x=time, y=actual)) + geom_point(color = "blue") + geom_line(color = "red", data = LR_1_results[LR_1_results$case == 15, ], aes(x = time, y = predicted)) + geom_line(color = "red", data = LR_1_results[LR_1_results$case == 15, ], aes(x = time, y = lbound), linetype = "dashed") + geom_line(color = "red", data = LR_1_results[LR_1_results$case == 15, ], aes(x = time, y = ubound), linetype = "dashed") + annotate("text", x = min(LR_1_results[LR_1_results$case == 15, ]$time), y = VB_k + 0.1, label = "Usure limite") + geom_hline(yintercept = VB_k)
plot_exp_12 <- ggplot(data=LR_1_results[LR_1_results$case == 12,], mapping=aes(x=time, y=actual)) + geom_point(color = "blue") + geom_line(color = "red", data = LR_1_results[LR_1_results$case == 12, ], aes(x = time, y = predicted)) + geom_line(color = "red", data = LR_1_results[LR_1_results$case == 12, ], aes(x = time, y = lbound), linetype = "dashed") + geom_line(color = "red", data = LR_1_results[LR_1_results$case == 12, ], aes(x = time, y = ubound), linetype = "dashed") + geom_hline(yintercept = VB_k)
plot_exp_11 <- ggplot(data=LR_1_results[LR_1_results$case == 11,], mapping=aes(x=time, y=actual)) + geom_point(color = "blue") + geom_line(color = "red", data = LR_1_results[LR_1_results$case == 11, ], aes(x = time, y = predicted)) + geom_line(color = "red", data = LR_1_results[LR_1_results$case == 11, ], aes(x = time, y = lbound), linetype = "dashed") + geom_line(color = "red", data = LR_1_results[LR_1_results$case == 11, ], aes(x = time, y = ubound), linetype = "dashed") + geom_hline(yintercept = VB_k)
plot_exp_16 <- ggplot(data=LR_1_results[LR_1_results$case == 16,], mapping=aes(x=time, y=actual)) + geom_point(color = "blue") + geom_line(color = "red", data = LR_1_results[LR_1_results$case == 16, ], aes(x = time, y = predicted)) + geom_line(color = "red", data = LR_1_results[LR_1_results$case == 16, ], aes(x = time, y = lbound), linetype = "dashed") + geom_line(color = "red", data = LR_1_results[LR_1_results$case == 16, ], aes(x = time, y = ubound), linetype = "dashed") + geom_hline(yintercept = VB_k)
ggarrange(plot_exp_11, plot_exp_12, plot_exp_15, plot_exp_16)

## an additional check: model 1's RMSE, MAE and average accuracy in the training set
real_train_predictions <- predict(LR_1, train_scaled) * (max(data_with_TF_coded$VB) - min(data_with_TF_coded$VB)) + min(data_with_TF_coded$VB)
real_train_time <- train_scaled$time * (max(data_with_TF_coded$time) - min(data_with_TF_coded$time)) + min(data_with_TF_coded$time)
real_train_VB <- train_scaled$VB * (max(data_with_TF_coded$VB) - min(data_with_TF_coded$VB)) + min(data_with_TF_coded$VB)
LR_1_results_train <- data.frame(time = real_train_time, case = train_scaled$case, actual = real_train_VB, predicted = real_train_predictions)
RMSE_LR_1_train <- rmse(actual = LR_1_results_train$actual, predicted = LR_1_results_train$predicted)

## 


# Model 2: with temporal features and process information
LR_2 <- lm(formula = VB ~ DOC + feed + material + RMS_smcAC + RMS_smcDC + RMS_vib_table + RMS_vib_spindle + RMS_AE_table + RMS_AE_spindle + var_smcAC + var_smcDC + var_vib_table + var_vib_spindle + var_AE_table + var_AE_spindle + max_smcAC + max_smcDC + max_vib_table + max_vib_spindle + max_AE_table + max_AE_spindle + skewness_smcAC + skewness_smcDC + skewness_vib_table + skewness_vib_spindle + skewness_AE_table + skewness_AE_spindle + kurtosis_smcAC + kurtosis_smcDC + kurtosis_vib_table + kurtosis_vib_spindle + kurtosis_AE_table + kurtosis_AE_spindle,
           data = train_scaled)
summary(LR_2)
# Model 3: with temporal features and time
LR_3 <- lm(formula = VB ~ time + RMS_smcAC + RMS_smcDC + RMS_vib_table + RMS_vib_spindle + RMS_AE_table + RMS_AE_spindle + var_smcAC + var_smcDC + var_vib_table + var_vib_spindle + var_AE_table + var_AE_spindle + max_smcAC + max_smcDC + max_vib_table + max_vib_spindle + max_AE_table + max_AE_spindle + skewness_smcAC + skewness_smcDC + skewness_vib_table + skewness_vib_spindle + skewness_AE_table + skewness_AE_spindle + kurtosis_smcAC + kurtosis_smDC + kurtosis_vib_table + kurtosis_vib_spindle + kurtosis_AE_table + kurtosis_AE_spindle,
           data = train_scaled)
summary(LR_3)
# Model 4: with temporal features only
LR_4 <- lm(formula = VB ~ RMS_smcAC + RMS_smcDC + RMS_vib_table + RMS_vib_spindle + RMS_AE_table + RMS_AE_spindle + var_smcAC + var_smcDC + var_vib_table + var_vib_spindle + var_AE_table + var_AE_spindle + max_smcAC + max_smcDC + max_vib_table + max_vib_spindle + max_AE_table + max_AE_spindle + skewness_smcAC + skewness_smcDC + skewness_vib_table + skewness_vib_spindle + skewness_AE_table + skewness_AE_spindle + kurtosis_smcAC + kurtosis_smDC + kurtosis_vib_table + kurtosis_vib_spindle + kurtosis_AE_table + kurtosis_AE_spindle,
           data = train_scaled)
summary(LR_4)
# Model 5: with process information and time
LR_5 <- lm(formula = VB ~ time + DOC + feed + material,
           data = train_scaled)
summary(LR_5)
# Model 6: with process information only
#LR_6 <- lm(formula = VB ~ DOC + feed + material,
 #          data = train_scaled)
#summary(LR_6)
# Model 7: with time only
LR_7 <- lm(formula = VB ~ time,
           data = train_scaled)
summary(LR_7)

####################################################### NEURAL NETWORKS FOR VB REGRESSION #########################################
set.seed(1234)
# Feedforward, fully connected NN (32 x 8 hidden units) for VB regression
NN_1 <- neuralnet(formula = VB ~ run + max_AE_spindle + skewness_vib_spindle + material,
                  data = train_scaled, hidden = c(32, 8), threshold = 0.01, linear.output = TRUE, act.fct = "tanh")
plot(NN_1)
NN_1$result.matrix

# Test the resulting output
temp_test <- subset(test_scaled, select = c("run", "max_AE_spindle", "skewness_vib_spindle", "material"))
head(temp_test)
nn.results <- compute(NN_1, temp_test)
results <- data.frame(actual = test_scaled$VB, prediction = nn.results$net.result)
print(results)

pr.nn_ <- nn.results$net.result * (max(data_with_temporal_features$VB) - min(data_with_temporal_features$VB)) + min(data_with_temporal_features$VB)
test.r <- (test_scaled$VB) * (max(data_with_temporal_features$VB) - min(data_with_temporal_features$VB)) + 
  min(data_with_temporal_features$VB)
MSE.nn_1 <- sum((test.r - pr.nn_)^2) / nrow(test_scaled)
RMSE_NN_1_test <- sqrt(MSE.nn_1)
#MSE.nn <- sum((test_scaled[,"VB"] - nn.results$net.result)^2) / nrow(test_scaled)

# Let's plot predicted vs. actual VB values for cases 11, 12, 15 and 16
plot(test_scaled$VB, nn.results$net.result, col = "red", 
     main = 'VB observées vs. VB prédites')
abline(0, 1, lwd = 2)

# A second feedforward, fully connected NN (3 hidden units) for VB regression
NN_2 <- neuralnet(formula = VB ~ run + max_AE_spindle + skewness_vib_spindle + material,
                  data = train_scaled, hidden = 3, threshold = 0.01, linear.output = TRUE, act.fct = "tanh")
plot(NN_2)
NN_2$result.matrix # Training loss is .21144 after 3990 steps

nn.results_2 <- compute(NN_2, temp_test)
results_2 <- data.frame(actual = test_scaled$VB, prediction = nn.results_2$net.result)
print(results)

pr.nn_2 <- nn.results_2$net.result * (max(data_with_temporal_features$VB) - min(data_with_temporal_features$VB)) + min(data_with_temporal_features$VB)
MSE.nn_2 <- sum((test.r - pr.nn_2)^2) / nrow(test_scaled)
RMSE_NN_2_test <- sqrt(MSE.nn_2)

# Let's plot predicted vs. actual VB values for cases 11, 12, 15 and 16
plot(test_scaled$VB, nn.results_2$net.result, col = "red", 
     main = 'VB observées vs. VB prédites par NN_2')
abline(0, 1, lwd = 2)

# A third feedforward, fully connected NN (2 hidden units) for VB regression
NN_3 <- neuralnet(formula = VB ~ run + max_AE_spindle + skewness_vib_spindle + material,
                  data = train_scaled, hidden = 2, threshold = 0.01, linear.output = TRUE, act.fct = "tanh")
plot(NN_3)
NN_3$result.matrix # Training loss is x after y steps

nn.results_3 <- compute(NN_3, temp_test)
results_3 <- data.frame(actual = test_scaled$VB, prediction = nn.results_2$net.result)
print(results_3)

pr.nn_3 <- nn.results_3$net.result * (max(data_with_temporal_features$VB) - min(data_with_temporal_features$VB)) + min(data_with_temporal_features$VB)
MSE.nn_3 <- sum((test.r - pr.nn_3)^2) / nrow(test_scaled)
RMSE_NN_3_test <- sqrt(MSE.nn_3)

#MSE.nn <- sum((test_scaled[,"VB"] - nn.results$net.result)^2) / nrow(test_scaled)
# Let's plot predicted vs. actual VB values for cases 11, 12, 15 and 16
plot(test_scaled$VB, nn.results_3$net.result, col = "red", 
     main = 'VB observées vs. VB prédites par NN_3')
abline(0, 1, lwd = 2)

# Let's take a closer look at each test experiment
## Case 11
results_with_time_and_case <- data.frame(time = test$time, case = test$case, actual = test.r, prediction_NN1 = nn.results$net.result, prediction_NN2 = nn.results_2$net.result, prediction_NN3 = nn.results_3$net.result)
plot_NN_exp_11 <- ggplot(data=results_with_time_and_case[results_with_time_and_case$case == 11,], mapping=aes(x=time, y=actual)) + geom_point(color = "blue") + geom_line(color = "red", data = results_with_time_and_case[results_with_time_and_case$case == 11, ], aes(x = time, y = prediction_NN1)) + geom_line(color = "orange", data = results_with_time_and_case[results_with_time_and_case$case == 11, ], aes(x = time, y = prediction_NN2)) + geom_line(color = "cyan", data = results_with_time_and_case[results_with_time_and_case$case == 11, ], aes(x = time, y = prediction_NN3)) + theme_set(theme_gray())
ggarrange(plot_NN_exp_11)

## Let's compute each NN's RMSE with respect to case 11
RMSE_NN_1_exp_11 <- rmse(actual = results_with_time_and_case[results_with_time_and_case$case == 11, ]$actual, predicted = results_with_time_and_case[results_with_time_and_case$case == 11, ]$prediction_NN1)
RMSE_NN_2_exp_11 <- rmse(actual = results_with_time_and_case[results_with_time_and_case$case == 11, ]$actual, predicted = results_with_time_and_case[results_with_time_and_case$case == 11, ]$prediction_NN2)
RMSE_NN_3_exp_11 <- rmse(actual = results_with_time_and_case[results_with_time_and_case$case == 11, ]$actual, predicted = results_with_time_and_case[results_with_time_and_case$case == 11, ]$prediction_NN3)

## Case 12
plot_NN_exp_12 <- ggplot(data=results_with_time_and_case[results_with_time_and_case$case == 12,], mapping=aes(x=time, y=actual)) + geom_point(color = "blue") + geom_line(color = "red", data = results_with_time_and_case[results_with_time_and_case$case == 12, ], aes(x = time, y = prediction_NN1)) + geom_line(color = "orange", data = results_with_time_and_case[results_with_time_and_case$case == 12, ], aes(x = time, y = prediction_NN2)) + geom_line(color = "cyan", data = results_with_time_and_case[results_with_time_and_case$case == 12, ], aes(x = time, y = prediction_NN3)) + theme_set(theme_gray())
ggarrange(plot_NN_exp_11, plot_NN_exp_12)

## Let's compute each NN's RMSE with respect to case 12
RMSE_NN_1_exp_12 <- rmse(actual = results_with_time_and_case[results_with_time_and_case$case == 12, ]$actual, predicted = results_with_time_and_case[results_with_time_and_case$case == 12, ]$prediction_NN1)
RMSE_NN_2_exp_12 <- rmse(actual = results_with_time_and_case[results_with_time_and_case$case == 12, ]$actual, predicted = results_with_time_and_case[results_with_time_and_case$case == 12, ]$prediction_NN2)
RMSE_NN_3_exp_12 <- rmse(actual = results_with_time_and_case[results_with_time_and_case$case == 12, ]$actual, predicted = results_with_time_and_case[results_with_time_and_case$case == 12, ]$prediction_NN3)

## Case 15
plot_NN_exp_15 <- ggplot(data=results_with_time_and_case[results_with_time_and_case$case == 15,], mapping=aes(x=time, y=actual)) + geom_point(color = "blue") + geom_line(color = "red", data = results_with_time_and_case[results_with_time_and_case$case == 15, ], aes(x = time, y = prediction_NN1)) + geom_line(color = "orange", data = results_with_time_and_case[results_with_time_and_case$case == 15, ], aes(x = time, y = prediction_NN2)) + geom_line(color = "cyan", data = results_with_time_and_case[results_with_time_and_case$case == 15, ], aes(x = time, y = prediction_NN3)) + theme_set(theme_gray())
ggarrange(plot_NN_exp_11, plot_NN_exp_12, plot_NN_exp_15)

## Let's compute each NN's RMSE with respect to case 15
RMSE_NN_1_exp_15 <- rmse(actual = results_with_time_and_case[results_with_time_and_case$case == 15, ]$actual, predicted = results_with_time_and_case[results_with_time_and_case$case == 15, ]$prediction_NN1)
RMSE_NN_2_exp_15 <- rmse(actual = results_with_time_and_case[results_with_time_and_case$case == 15, ]$actual, predicted = results_with_time_and_case[results_with_time_and_case$case == 15, ]$prediction_NN2)
RMSE_NN_3_exp_15 <- rmse(actual = results_with_time_and_case[results_with_time_and_case$case == 15, ]$actual, predicted = results_with_time_and_case[results_with_time_and_case$case == 15, ]$prediction_NN3)

## Case 16
plot_NN_exp_16 <- ggplot(data=test[test$case == 16,], mapping=aes(x=time, y=VB)) + geom_point(color = "blue") + geom_line(color = "red", data = results_with_time_and_case, aes(x = time, y = prediction_NN1)) + geom_line(color = "orange", data = results_with_time_and_case, aes(x = time, y = prediction_NN2)) + geom_line(color = "cyan", data = results_with_time_and_case, aes(x = time, y = prediction_NN3)) + theme_set(theme_gray())
ggarrange(plot_NN_exp_11, plot_NN_exp_12, plot_NN_exp_15, plot_NN_exp_16)

## Let's compute each NN's RMSE with respect to case 15
RMSE_NN_1_exp_16 <- rmse(actual = results_with_time_and_case[results_with_time_and_case$case == 16, ]$actual, predicted = results_with_time_and_case[results_with_time_and_case$case == 16, ]$prediction_NN1)
RMSE_NN_2_exp_16 <- rmse(actual = results_with_time_and_case[results_with_time_and_case$case == 16, ]$actual, predicted = results_with_time_and_case[results_with_time_and_case$case == 16, ]$prediction_NN2)
RMSE_NN_3_exp_16 <- rmse(actual = results_with_time_and_case[results_with_time_and_case$case == 16, ]$actual, predicted = results_with_time_and_case[results_with_time_and_case$case == 16, ]$prediction_NN3)

### SUPPORT VECTOR REGRESSION FOR VB ###
SVM_1 <- svm(formula = VB ~ run + max_AE_spindle + skewness_vib_spindle + material,
             data = train_scaled)
real_predictions_SVM_1 <- predict(SVM_1, test_scaled) * (max(data_with_TF_coded$VB) - min(data_with_TF_coded$VB)) + min(data_with_TF_coded$VB)
SVM_1_results <- data.frame(run = real_run, time = real_time, case = test_scaled$case, actual = real_VB, predicted = real_predictions_SVM_1)

# Let's plot predicted vs. actual VB values for cases 11, 12, 15 and 16
plot(SVM_1_results$actual, SVM_1_results$predicted, col = "red", 
     main = 'VB observées vs. VB prédites par SVM_1')
abline(0, 1, lwd = 2)

## Let's compute SVM_1's RMSE, MAE and average accuracy in cases 11, 12, 15 and 16
RMSE_SVM_1_exp_11 <- rmse(actual = SVM_1_results[SVM_1_results$case == 11, "actual"], predicted = SVM_1_results[SVM_1_results$case == 11, "predicted"])
RMSE_SVM_1_exp_12 <- rmse(actual = SVM_1_results[SVM_1_results$case == 12, "actual"], predicted = SVM_1_results[SVM_1_results$case == 12, "predicted"])
RMSE_SVM_1_exp_15 <- rmse(actual = SVM_1_results[SVM_1_results$case == 15, "actual"], predicted = SVM_1_results[SVM_1_results$case == 15, "predicted"])
RMSE_SVM_1_exp_16 <- rmse(actual = SVM_1_results[SVM_1_results$case == 16, "actual"], predicted = SVM_1_results[SVM_1_results$case == 16, "predicted"])
RMSE_SVM_1_test <- rmse(actual = SVM_1_results$actual, predicted = SVM_1_results$predicted)
MAE_SVM_1_exp_11 <- rmse(actual = SVM_1_results[SVM_1_results$case == 11, "actual"], predicted = SVM_1_results[SVM_1_results$case == 11, "predicted"])
MAE_SVM_1_exp_12 <- rmse(actual = SVM_1_results[SVM_1_results$case == 12, "actual"], predicted = SVM_1_results[SVM_1_results$case == 12, "predicted"])
MAE_SVM_1_exp_15 <- rmse(actual = SVM_1_results[SVM_1_results$case == 15, "actual"], predicted = SVM_1_results[SVM_1_results$case == 15, "predicted"])
MAE_SVM_1_exp_16 <- rmse(actual = SVM_1_results[SVM_1_results$case == 16, "actual"], predicted = SVM_1_results[SVM_1_results$case == 16, "predicted"])
MAE_SVM_1_test <- mae(actual = SVM_1_results$actual, predicted = SVM_1_results$predicted)
avg_accuracy_SVM_1_exp_11 <- MAE_SVM_1_exp_11 / mean(SVM_1_results[SVM_1_results$case == 11, "actual"])
avg_accuracy_SVM_1_exp_12 <- MAE_SVM_1_exp_12 / mean(SVM_1_results[SVM_1_results$case == 12, "actual"])
avg_accuracy_SVM_1_exp_15 <- MAE_SVM_1_exp_15 / mean(SVM_1_results[SVM_1_results$case == 15, "actual"])
avg_accuracy_SVM_1_exp_16 <- MAE_SVM_1_exp_16 / mean(SVM_1_results[SVM_1_results$case == 16, "actual"])
avg_accuracy_SVM_1_test <- MAE_SVM_1_test / mean(SVM_1_results$actual)

## Let's compare the actual to the predicted increase in flank wear in cases 11, 12, 15 and 16
plot_exp_15 <- ggplot(data=SVM_1_results[SVM_1_results$case == 15,], mapping=aes(x=time, y=actual)) + geom_point(color = "blue") + geom_line(color = "red", data = SVM_1_results[SVM_1_results$case == 15, ], aes(x = time, y = predicted)) + geom_hline(yintercept = VB_k)
plot_exp_12 <- ggplot(data=SVM_1_results[SVM_1_results$case == 12,], mapping=aes(x=time, y=actual)) + geom_point(color = "blue") + geom_line(color = "red", data = SVM_1_results[SVM_1_results$case == 12, ], aes(x = time, y = predicted)) + geom_hline(yintercept = VB_k)
plot_exp_11 <- ggplot(data=SVM_1_results[SVM_1_results$case == 11,], mapping=aes(x=time, y=actual)) + geom_point(color = "blue") + geom_line(color = "red", data = SVM_1_results[SVM_1_results$case == 11, ], aes(x = time, y = predicted)) + geom_hline(yintercept = VB_k)
plot_exp_16 <- ggplot(data=SVM_1_results[SVM_1_results$case == 16,], mapping=aes(x=time, y=actual)) + geom_point(color = "blue") + geom_line(color = "red", data = SVM_1_results[SVM_1_results$case == 16, ], aes(x = time, y = predicted)) + geom_hline(yintercept = VB_k)
ggarrange(plot_exp_11, plot_exp_12, plot_exp_15, plot_exp_16)

## an additional check: SVM_1's RMSE, MAE and average accuracy in the training set
SVM_1_results_train <- data.frame(time = train_scaled$time, case = train_scaled$case, actual = train_scaled$VB, predicted = predict(SVM_1, train_scaled))
RMSE_SVM_1_train <- rmse(actual = train_scaled$VB, predicted = SVM_1_results_train$predicted)

# perform a grid search 
# (this might take a few seconds, adjust how fine of grid if taking too long)
tuneResult1 <- tune(svm, VB ~ run + max_AE_spindle + skewness_vib_spindle + material,
                    data = train_scaled,
                    ranges = list(epsilon = seq(0,1,0.1), cost = 2^(seq(0.5,8,.5)))
)

# Map tuning results
plot(tuneResult1)
print(tuneResult1)
## Best parameters: epsilon = 0, cost = 2.828427

SVM_2 <- tuneResult1$best.model
real_predictions_SVM_2 <- predict(SVM_2, test_scaled) * (max(data_with_TF_coded$VB) - min(data_with_TF_coded$VB)) + min(data_with_TF_coded$VB)
SVM_2_results <- data.frame(run = real_run, time = real_time, case = test_scaled$case, actual = real_VB, predicted = real_predictions_SVM_2)



# Let's plot smcAC sensor data for the first run of experiment 2
#smcAC_run_1 <- c(experiments_2_10[1, 8:9007])
#smcAC_run_1_indices <- c(1:9000)
#smcAC_run_1_points <- data.frame(smcAC_run_1_indices, smcAC_run_1)
#ggplot(data=smcAC_run_1)

#Data Analysis 3 
#A3
#Abigail Chen
################################################################################
#Setting the environment########################################################
################################################################################
rm(list=ls())

#Please change dir to your own and unzip bisnode firm panel data in data/raw folder

#Loading the libraries

library(caret)
library(cowplot)
library(glmnet)
library(gmodels) 
library(haven)
library(Hmisc)
library(kableExtra)
library(lspline)
library(margins)
library(modelsummary)
library(partykit)
library(pROC)
library(purrr)
library(ranger)
library(rattle)
library(rpart)
library(rpart.plot)
library(sandwich)
library(skimr)
library(viridis)


#Get working directory
getwd()

################################################################################
#Setting the directory##########################################################
################################################################################

source("/Users/abigailchristinechen/da3/a3/da_helper_functions.R")

path <-  "/Users/abigailchristinechen/da3/a3/"
data_in <- paste0(path,"data/clean/")
data_out <- data_in
output <- paste0(path,"output/")
create_output_if_doesnt_exist(output)

################################################################################
#Loading the data###############################################################
################################################################################

#10,528 Observations
#117 Variables
data <- readRDS(paste0(data_in,"bisnode_firms_clean.rds"))

glimpse( data )

################################################################################
#Summary########################################################################
################################################################################

skimr::skim(data)
datasummary_skim(data, type="categorical")

################################################################################
#Define variable################################################################
################################################################################

#Main firm variables
rawvars <-  c("curr_assets", "curr_liab", "extra_exp", "extra_inc", "extra_profit_loss", "fixed_assets",
              "inc_bef_tax", "intang_assets", "inventories", "liq_assets", "material_exp", "personnel_exp",
              "profit_loss_year", "sales", "share_eq", "subscribed_cap")

#Further financial variables
qualityvars <- c("balsheet_flag", "balsheet_length", "balsheet_notfullyear")
engvar <- c("total_assets_bs", "fixed_assets_bs", "liq_assets_bs", "curr_assets_bs",
            "share_eq_bs", "subscribed_cap_bs", "intang_assets_bs", "extra_exp_pl",
            "extra_inc_pl", "extra_profit_loss_pl", "inc_bef_tax_pl", "inventories_pl",
            "material_exp_pl", "profit_loss_year_pl", "personnel_exp_pl")
engvar <- c("total_assets_bs", "fixed_assets_bs", "liq_assets_bs", "curr_assets_bs",
            "share_eq_bs", "subscribed_cap_bs", "intang_assets_bs", "extra_exp_pl",
            "extra_inc_pl", "extra_profit_loss_pl", "inc_bef_tax_pl", "inventories_pl",
            "material_exp_pl", "profit_loss_year_pl", "personnel_exp_pl")
engvar2 <- c("extra_profit_loss_pl_quad", "inc_bef_tax_pl_quad",
             "profit_loss_year_pl_quad", "share_eq_bs_quad")

#Flag variables
engvar3 <- c(grep("*flag_low$", names(data), value = TRUE),
             grep("*flag_high$", names(data), value = TRUE),
             grep("*flag_error$", names(data), value = TRUE),
             grep("*flag_zero$", names(data), value = TRUE))

#Growth variables
d1 <-  c("d1_sales_mil_log_mod", "d1_sales_mil_log_mod_sq",
         "flag_low_d1_sales_mil_log", "flag_high_d1_sales_mil_log")

#Human capital related variables
hr <- c("female", "ceo_age", "flag_high_ceo_age", "flag_low_ceo_age",
        "flag_miss_ceo_age", "ceo_count", "labor_avg_mod",
        "flag_miss_labor_avg", "foreign_management")

#Firms history related variables
firm <- c("age", "age2", "new", "ind2_cat", "m_region_loc", "urban_m")


#Interactions for logit, LASSO
interactions1 <- c("ind2_cat*age", "ind2_cat*age2",
                   "ind2_cat*d1_sales_mil_log_mod", "ind2_cat*sales_mil_log",
                   "ind2_cat*ceo_age", "ind2_cat*foreign_management",
                   "ind2_cat*female",   "ind2_cat*urban_m", "ind2_cat*labor_avg_mod")
interactions2 <- c("sales_mil_log*age", "sales_mil_log*female",
                   "sales_mil_log*profit_loss_year_pl", "sales_mil_log*foreign_management")


#Simple logit models 
X1 <- c("sales_mil_log", "sales_mil_log_sq", "d1_sales_mil_log_mod", "profit_loss_year_pl", "ind2_cat")
X2 <- c("sales_mil_log", "sales_mil_log_sq", "d1_sales_mil_log_mod", "profit_loss_year_pl", "fixed_assets_bs","share_eq_bs","curr_liab_bs ",   "curr_liab_bs_flag_high ", "curr_liab_bs_flag_error",  "age","foreign_management" , "ind2_cat")
X3 <- c("sales_mil_log", "sales_mil_log_sq", firm, engvar, d1)
X4 <- c("sales_mil_log", "sales_mil_log_sq", firm, engvar, engvar2, engvar3, d1, hr)
X5 <- c("sales_mil_log", "sales_mil_log_sq", firm, engvar, engvar2, engvar3, d1, hr, interactions1, interactions2)

#Logit + LASSO
logitvars <- c("sales_mil_log", "sales_mil_log_sq", engvar, engvar2, engvar3, d1, hr, firm, qualityvars, interactions1, interactions2)

#Cart + RF (no interactions, no modified features)
rfvars  <-  c("sales_mil", "d1_sales_mil_log", rawvars, hr, firm)
rfvars  <-  c("sales_mil", "d1_sales_mil_log", rawvars, hr, firm, qualityvars)


#Check X1
ols_modelx1 <- feols(formula(paste0("fast_growth ~", paste0(X1, collapse = " + "))),
                  data = data)
summary(ols_modelx1)

#Logit Model
glm_modelx1 <- glm(formula(paste0("fast_growth ~", paste0(X1, collapse = " + "))),
                   data = data, family = "binomial")
summary(glm_modelx1)

#Check X2
glm_modelx2 <- glm(formula(paste0("fast_growth ~", paste0(X2, collapse = " + "))),
                   data = data, family = "binomial")
summary(glm_modelx2)



#Note:
#With Logit we need to calculate average marginal effects (dy/dx)
#to be able to interpret the coefficients (under some assumptions...)
#vce="none" makes it run much faster, here we do not need variances
#but if you do need it (e.g. prediction intervals -> take the time!)
mx2 <- margins(glm_modelx2, vce = "none")

sum_table <- summary(glm_modelx2) %>%
  coef() %>%
  as.data.frame() %>%
  select(Estimate) %>%
  mutate(factor = row.names(.)) %>%
  merge(summary(mx2)[,c("factor","AME")])

sum_table

ols_modelx4 <- lm(formula(paste0("fast_growth ~", paste0(X4, collapse = " + "))),
                  data = data)
summary(ols_modelx4)

glm_modelx4 <- glm(formula(paste0("fast_growth ~", paste0(X4, collapse = " + "))),
                   data = data, family = "binomial")
summary(glm_modelx4)

m <- margins(glm_modelx4, vce = "none")

sum_table2 <- summary(glm_modelx4) %>%
  coef() %>%
  as.data.frame() %>%
  select(Estimate, `Std. Error`) %>%
  mutate(factor = row.names(.)) %>%
  merge(summary(m)[,c("factor","AME")])

sum_table2


################################################################################
#Model Building#################################################################
################################################################################

#STEP 0)
#Separate data sets
#Train and holdout samples

set.seed(2022)

train_indices <- as.integer(createDataPartition(data$fast_growth, p = 0.8, list = FALSE))
data_train <- data[train_indices, ]
data_holdout <- data[-train_indices, ]

dim(data_train)
dim(data_holdout)

Hmisc::describe(data$fast_growth_f)
Hmisc::describe(data_train$fast_growth_f)
Hmisc::describe(data_holdout
                $fast_growth_f)

#Step 1) Predict probabilities 
#with logit + logit and LASSO models
#use CV


#5 fold cross-validation:
#check the summary function
train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummaryExtended,
  savePredictions = TRUE
)


#Prob. LOGIT models ############################################################
logit_model_vars <- list("X1" = X1, "X2" = X2, "X3" = X3, "X4" = X4, "X5" = X5)

CV_RMSE_folds <- list()
logit_models <- list()

for (model_name in names(logit_model_vars)) {
  
  features <- logit_model_vars[[model_name]]
  
  set.seed(2021)
  glm_model <- train(
    formula(paste0("fast_growth_f ~", paste0(features, collapse = " + "))),
    method = "glm",
    data = data_train,
    family = binomial,
    trControl = train_control
  )
  
  logit_models[[model_name]] <- glm_model
  # Calculate RMSE on test for each fold
  CV_RMSE_folds[[model_name]] <- glm_model$resample[,c("Resample", "RMSE")]
  }

#Logit + LASSO #################################################################

# Set lambda parameters to check
lambda <- 10^seq(-1, -4, length = 10)
grid <- expand.grid("alpha" = 1, lambda = lambda)

# Estimate logit + LASSO with 5-fold CV to find lambda
set.seed(2022)
system.time({
  logit_lasso_model <- train(
    formula(paste0("fast_growth_f ~", paste0(logitvars, collapse = " + "))),
    data = data_train,
    method = "glmnet",
    preProcess = c("center", "scale"),
    family = "binomial",
    trControl = train_control,
    tuneGrid = grid,
    na.action=na.exclude
  )
})

#Saving  the results
tuned_logit_lasso_model <- logit_lasso_model$finalModel
best_lambda <- logit_lasso_model$bestTune$lambda
logit_models[["LASSO"]] <- logit_lasso_model
lasso_coeffs <- as.matrix(coef(tuned_logit_lasso_model, best_lambda))
CV_RMSE_folds[["LASSO"]] <- logit_lasso_model$resample[,c("Resample", "RMSE")]


#STEP 2)
#Calibration Curve, Confusion Matrix,
#ROC, AUC, 

#Draw ROC Curve and calculate AUC for each folds
CV_AUC_folds <- list()

for (model_name in names(logit_models)) {
  
  auc <- list()
  model <- logit_models[[model_name]]
  for (fold in c("Fold1", "Fold2", "Fold3", "Fold4", "Fold5")) {
    cv_fold <-
      model$pred %>%
      filter(Resample == fold)
    
    roc_obj <- roc(cv_fold$obs, cv_fold$fast_growth)
    auc[[fold]] <- as.numeric(roc_obj$auc)
  }
  
  CV_AUC_folds[[model_name]] <- data.frame("Resample" = names(auc),
                                           "AUC" = unlist(auc))
}


#for each model: average RMSE and average AUC for each models
CV_RMSE <- list()
CV_AUC <- list()

for (model_name in names(logit_models)) {
  CV_RMSE[[model_name]] <- mean(CV_RMSE_folds[[model_name]]$RMSE)
  CV_AUC[[model_name]] <- mean(CV_AUC_folds[[model_name]]$AUC)
}

#We have 6 models, (5 logit and the logit lasso). For each we have a 5-CV RMSE and AUC.
# We pick our preferred model based on that.

nvars <- lapply(logit_models, FUN = function(x) length(x$coefnames))
# quick adjustment for LASSO
nvars[["LASSO"]] <- sum(lasso_coeffs != 0)

logit_summary1 <- data.frame("Number of predictors" = unlist(nvars),
                             "CV RMSE" = unlist(CV_RMSE),
                             "CV AUC" = unlist(CV_AUC))

#Use summary for average RMSE and AUC for each model on the test sample
logit_summary1

#Estimate RMSE on holdout sample
#Take best model and estimate RMSE on holdout###################################

best_logit_no_loss <- logit_models[["X3"]]

logit_predicted_probabilities_holdout <- predict(best_logit_no_loss, newdata = data_holdout, type = "prob")
data_holdout[,"best_logit_no_loss_pred"] <- logit_predicted_probabilities_holdout[,"fast_growth"]
RMSE(data_holdout[, "best_logit_no_loss_pred", drop=TRUE], data_holdout$fast_growth)

#Discrete ROC (with thresholds in steps) on holdout

# Confusion Matrix, with (arbitrarily) chosen threshold(s)######################

# fast_growth: the threshold 0.5 is used to convert probabilities to binary classes
logit_class_prediction <- predict(best_logit_no_loss, newdata = data_holdout)
summary(logit_class_prediction)

# Confusion matrix: summarize different type of errors and successfully predicted cases
# positive = "yes": explicitly specify the positive case
cm_object1 <- confusionMatrix(logit_class_prediction, data_holdout$fast_growth_f, positive = "fast_growth")
cm_object1
cm1 <- cm_object1$table
cm1


# we can apply different thresholds:
# 0.5 same as before

# c) Visualize ROC (with thresholds in steps) on holdout
#     what if we want to compare multiple thresholds?
#   Get AUC - how good our model is in terms of classification error?

thresholds <- seq(0.05, 0.75, by = 0.05)
holdout_prediction <-
  ifelse(data_holdout$best_logit_no_loss_pred < 0.5, "no_fast_growth", "fast_growth") %>%
  factor(levels = c("no_fast_growth", "fast_growth"))
cm_object1b <- confusionMatrix(holdout_prediction,data_holdout$fast_growth_f)
cm1b <- cm_object1b$table
cm1b

# a sensible choice: mean of predicted probabilities
mean_predicted_fast_growth_prob <- mean(data_holdout$best_logit_no_loss_pred)
mean_predicted_fast_growth_prob
holdout_prediction <-
  ifelse(data_holdout$best_logit_no_loss_pred < mean_predicted_fast_growth_prob, "no_fast_growth", "fast_growth") %>%
  factor(levels = c("no_fast_growth", "fast_growth"))
cm_object2 <- confusionMatrix(holdout_prediction,data_holdout$fast_growth_f)
cm2 <- cm_object2$table
cm2

#Calibration curve##############################################################
# how well do estimated vs actual event probabilities relate to each other?


create_calibration_plot(data_holdout, 
                        file_name = "ch17-figure-1-logit-m4-calibration", 
                        prob_var = "best_logit_no_loss_pred", 
                        actual_var = "fast_growth",
                        n_bins = 10)
# We have a loss function




# pre allocate lists
cm <- list()
true_positive_rates <- c()
false_positive_rates <- c()
for (thr in thresholds) {
  holdout_prediction <- ifelse(data_holdout[,"best_logit_no_loss_pred"] < thr, "no_fast_growth", "fast_growth") %>%
    factor(levels = c("no_fast_growth", "fast_growth"))
  cm_thr <- confusionMatrix(holdout_prediction,data_holdout$fast_growth_f)$table
  cm[[as.character(thr)]] <- cm_thr
  true_positive_rates <- c(true_positive_rates, cm_thr["fast_growth", "fast_growth"] /
                             (cm_thr["fast_growth", "fast_growth"] + cm_thr["no_fast_growth", "fast_growth"]))
  false_positive_rates <- c(false_positive_rates, cm_thr["fast_growth", "no_fast_growth"] /
                              (cm_thr["fast_growth", "no_fast_growth"] + cm_thr["no_fast_growth", "no_fast_growth"]))
}

# create a tibble for results
tpr_fpr_for_thresholds <- tibble("threshold" = thresholds,
                                 "true_positive_rate" = true_positive_rates,
                                 "false_positive_rate" = false_positive_rates)

#Plot discrete ROC
discrete_roc_plot <- ggplot(
  data = tpr_fpr_for_thresholds,
  aes(x = false_positive_rate, y = true_positive_rate, color = threshold)) +
  labs(x = "False positive rate (1 - Specificity)", y = "True positive rate (Sensitivity)") +
  geom_point(size=2, alpha=0.8) +
  scale_color_viridis(option = "D", direction = -1) +
  scale_x_continuous(expand = c(0.01,0.01), limit=c(0,1), breaks = seq(0,1,0.1)) +
  scale_y_continuous(expand = c(0.01,0.01), limit=c(0,1), breaks = seq(0,1,0.1)) +
  theme_wsj() + scale_colour_wsj("colors6") +
  theme(legend.position ="right") +
  theme(legend.title = element_text(size = 4), 
        legend.text = element_text(size = 4),
        legend.key.size = unit(.4, "cm")) 
discrete_roc_plot
save_fig("ch17-figure-2a-roc-discrete", output, "small")

# Or with a fairly easy commands, we can plot, the
# continuous ROC on holdout with Logit 4########################################
roc_obj_holdout <- roc(data_holdout$fast_growth, data_holdout$best_logit_no_loss_pred, quiet = T)
# use aux function
createRocPlot(roc_obj_holdout, "best_logit_no_loss_roc_plot_holdout")

# and quantify the AUC (Area Under the (ROC) Curve)
roc_obj_holdout$auc

# Confusion table with different tresholds ----------------------------------------------------------

# fast_growth: the threshold 0.5 is used to convert probabilities to binary classes
logit_class_prediction <- predict(best_logit_no_loss, newdata = data_holdout)
summary(logit_class_prediction)

# confusion matrix: summarize different type of errors and successfully predicted cases
# positive = "yes": explicitly specify the positive case
cm_object1 <- confusionMatrix(logit_class_prediction, data_holdout$fast_growth_f, positive = "fast_growth")
cm1 <- cm_object1$table
cm1


# Introduce loss function
# relative cost of of a false negative classification (as compared with a false positive classification)
FP=1
FN=10
cost = FN/FP
# the prevalence, or the proportion of cases in the population (n.cases/(n.controls+n.cases))
prevelance = sum(data_train$fast_growth)/length(data_train$fast_growth)

# Draw ROC Curve and find optimal threshold WITH loss function

best_tresholds <- list()
expected_loss <- list()
logit_cv_rocs <- list()
logit_cv_threshold <- list()
logit_cv_expected_loss <- list()

# Iterate through:
#  a) models
#  b) Folds
for (model_name in names(logit_models)) {
  
  model <- logit_models[[model_name]]
  colname <- paste0(model_name,"_prediction")
  
  best_tresholds_cv <- list()
  expected_loss_cv <- list()
  
  for (fold in c("Fold1", "Fold2", "Fold3", "Fold4", "Fold5")) {
    cv_fold <-
      model$pred %>%
      filter(Resample == fold)
    
    roc_obj <- roc(cv_fold$obs, cv_fold$fast_growth)
    best_treshold <- coords(roc_obj, "best", ret="all", transpose = FALSE,
                            best.method="youden", best.weights=c(cost, prevelance))
    # save best treshold for each fold and save the expected loss value
    best_tresholds_cv[[fold]] <- best_treshold$threshold
    expected_loss_cv[[fold]] <- (best_treshold$fp*FP + best_treshold$fn*FN)/length(cv_fold$fast_growth)
  }

  # average
  best_tresholds[[model_name]] <- mean(unlist(best_tresholds_cv))
  expected_loss[[model_name]] <- mean(unlist(expected_loss_cv))
  
  # for fold #5
  logit_cv_rocs[[model_name]] <- roc_obj
  logit_cv_threshold[[model_name]] <- best_treshold
  logit_cv_expected_loss[[model_name]] <- expected_loss_cv[[fold]]
  
}

logit_summary2 <- data.frame("Avg of optimal thresholds" = unlist(best_tresholds),
                             "Threshold for Fold5" = sapply(logit_cv_threshold, function(x) {x$threshold}),
                             "Avg expected loss" = unlist(expected_loss),
                             "Expected loss for Fold5" = unlist(logit_cv_expected_loss))

kable(x = logit_summary2, format = "latex", booktabs=TRUE,  digits = 3, row.names = TRUE,
      linesep = "", col.names = c("Avg of optimal thresholds","Threshold for fold #5",
                                  "Avg expected loss","Expected loss for fold #5")) %>%
  cat(.,file= paste0(output, "logit_summary1.tex"))

# Create plots based on Fold5 in CV#############################################

for (model_name in names(logit_cv_rocs)) {
  
  r <- logit_cv_rocs[[model_name]]
  best_coords <- logit_cv_threshold[[model_name]]
  createLossPlot(r, best_coords,
                 paste0(model_name, "_loss_plot"))
  createRocPlotWithOptimal(r, best_coords,
                           paste0(model_name, "_roc_plot"))
}


# Pick best model based on average expected loss ----------------------------------
# Pick best model based on average expected loss
# Calculate the expected loss on holdout sampl
best_logit_with_loss <- logit_models[["X4"]]
best_logit_optimal_treshold <- best_tresholds[["X4"]]


# get the ROC properties
r <- logit_cv_rocs[["X4"]]
# get coordinates and properties of the choosen threshold
best_coords <- logit_cv_threshold[["X4"]]
# plot for Loss function
createLossPlot(r, best_coords,
               paste0("X4", "_loss_plot"))
# Plot for optimal ROC
createRocPlotWithOptimal(r, best_coords,
                         paste0("X4", "_roc_plot"))

# Predict the probabilities on holdout
logit_predicted_probabilities_holdout <- predict(best_logit_with_loss, newdata = data_holdout, type = "prob")
data_holdout[,"best_logit_with_loss_pred"] <- logit_predicted_probabilities_holdout[,"fast_growth"]

# ROC curve on holdout
roc_obj_holdout <- roc(data_holdout$fast_growth, data_holdout[, "best_logit_with_loss_pred", drop=TRUE])

# Get expected loss on holdout
holdout_threshold <- coords(roc_obj_holdout, x = best_logit_optimal_treshold, input= "threshold",
                           ret="all", transpose = FALSE)
# Calculate the expected loss on holdout sample
expected_loss_holdout <- (holdout_threshold$fp*FP + holdout_threshold$fn*FN)/length(data_holdout$fast_growth)
expected_loss_holdout

# Confusion table on holdout with optimal threshold
holdout_prediction <-
  ifelse(data_holdout$best_logit_with_loss_pred < best_logit_optimal_treshold, "no_fast_growth", "fast_growth") %>%
  factor(levels = c("no_fast_growth", "fast_growth"))
cm_object3 <- confusionMatrix(holdout_prediction,data_holdout$fast_growth_f)
cm3 <- cm_object3$table
cm3

# in pctg
round( cm3 / sum(cm3) * 100 , 1 )


#PREDICTION WITH RANDOM FOREST#################################################

### 
# warm up with CART
data_for_graph <- data_train
levels(data_for_graph$fast_growth_f) <- list("stay" = "no_fast_growth", "exit" = "fast_growth")

# First a simple CART (with pre-set cp and minbucket)
set.seed(2022)
rf_for_graph <-
  rpart(
    formula = fast_growth_f ~ sales_mil + profit_loss_year+ foreign_management,
    data = data_for_graph,
    control = rpart.control(cp = 0.0028, minbucket = 100)
  )

rpart.plot(rf_for_graph, tweak=1, digits=2, extra=107, under = TRUE)
save_tree_plot(rf_for_graph, "tree_plot", output, "small", tweak=1)

#############

data_for_graph <- data_train
levels(data_for_graph$fast_growth) <- list("stay" = "no_fast_growth", "exit" = "fast_growth")


#################################################
# A) Probability forest
#     Split by gini, ratio of 1's in each tree, 
#      and average over trees


# 5 fold cross-validation
train_control <- trainControl(
  method = "cv",
  n = 5,
  classProbs = TRUE, # same as probability = TRUE in ranger
  summaryFunction = twoClassSummaryExtended,
  savePredictions = TRUE
)
train_control$verboseIter <- TRUE

#Probability forest#############################################################
# Split by gini, ratio of 1's in each tree, average over trees

# 5 fold cross-validation

train_control <- trainControl(
  method = "cv",
  n = 5,
  classProbs = TRUE, # same as probability = TRUE in ranger
  summaryFunction = twoClassSummaryExtended,
  savePredictions = TRUE
)
train_control$verboseIter <- TRUE

# Tuning parameters -> now only check for one setup, 
#     but you can play around with the rest, which is uncommented
tune_grid <- expand.grid(
  .mtry = c(5, 6, 7),
  .splitrule = "gini",
  .min.node.size = c(10, 15)
)
# By default ranger understoods that the outcome is binary, 
#   thus needs to use 'gini index' to decide split rule
# getModelInfo("ranger")
set.seed(123456)
rf_model_p <- train(
  formula(paste0("fast_growth_f ~ ", paste0(rfvars , collapse = " + "))),
  method = "ranger",
  data = data_train,
  tuneGrid = tune_grid,
  trControl = train_control
)

rf_model_p$results

best_mtry <- rf_model_p$bestTune$mtry
best_min_node_size <- rf_model_p$bestTune$min.node.size

# Get average (ie over the folds) RMSE and AUC ------------------------------------
CV_RMSE_folds[["rf_p"]] <- rf_model_p$resample[,c("Resample", "RMSE")]

auc <- list()
for (fold in c("Fold1", "Fold2", "Fold3", "Fold4", "Fold5")) {
  cv_fold <-
    rf_model_p$pred %>%
    filter(Resample == fold)
  
  roc_obj <- roc(cv_fold$obs, cv_fold$fast_growth)
  auc[[fold]] <- as.numeric(roc_obj$auc)
}
CV_AUC_folds[["rf_p"]] <- data.frame("Resample" = names(auc),
                                     "AUC" = unlist(auc))

CV_RMSE[["rf_p"]] <- mean(CV_RMSE_folds[["rf_p"]]$RMSE)
CV_AUC[["rf_p"]] <- mean(CV_AUC_folds[["rf_p"]]$AUC)

# Now use loss function and search for best thresholds and expected loss over folds -----
best_tresholds_cv <- list()
expected_loss_cv <- list()

for (fold in c("Fold1", "Fold2", "Fold3", "Fold4", "Fold5")) {
  cv_fold <-
    rf_model_p$pred %>%
    filter(mtry == best_mtry,
           min.node.size == best_min_node_size,
           Resample == fold)
  
  roc_obj <- roc(cv_fold$obs, cv_fold$fast_growth)
  best_treshold <- coords(roc_obj, "best", ret="all", transpose = FALSE,
                          best.method="youden", best.weights=c(cost, prevelance))
  best_tresholds_cv[[fold]] <- best_treshold$threshold
  expected_loss_cv[[fold]] <- (best_treshold$fp*FP + best_treshold$fn*FN)/length(cv_fold$fast_growth)
}

# Now use loss function and search for best thresholds and expected loss over folds 
best_thresholds_cv <- list()
expected_loss_cv <- list()

for (fold in c("Fold1", "Fold2", "Fold3", "Fold4", "Fold5")) {
  cv_fold <-
    rf_model_p$pred %>%
    filter(mtry == best_mtry,
           min.node.size == best_min_node_size,
           Resample == fold)
  
  roc_obj <- roc(cv_fold$obs, cv_fold$fast_growth)
  best_threshold <- coords(roc_obj, "best", ret="all", transpose = FALSE,
                          best.method="youden", best.weights=c(cost, prevelance))
  best_thresholds_cv[[fold]] <- best_threshold$threshold
  expected_loss_cv[[fold]] <- (best_threshold$fp*FP + best_threshold$fn*FN)/length(cv_fold$fast_growth)
}

# average
best_thresholds[["rf_p"]] <- mean(unlist(best_thresholds_cv))
expected_loss[["rf_p"]] <- mean(unlist(expected_loss_cv))


rf_summary <- data.frame("CV RMSE" = CV_RMSE[["rf_p"]],
                         "CV AUC" = CV_AUC[["rf_p"]],
                         "Avg of optimal thresholds" = best_threshold[["rf_p"]],
                         "Threshold for Fold5" = best_threshold$threshold,
                         "Avg expected loss" = expected_loss[["rf_p"]],
                         "Expected loss for Fold5" = expected_loss_cv[[fold]])

rf_summary 


###
# Create plots - this is for Fold5

createLossPlot(roc_obj, best_threshold, "rf_p_loss_plot")
createRocPlotWithOptimal(roc_obj, best_threshold, "rf_p_roc_plot")


#Take model to holdout and estimate RMSE, AUC and expected loss
rf_predicted_probabilities_holdout <- predict(rf_model_p, newdata = data_holdout, type = "prob")
data_holdout$rf_p_prediction <- rf_predicted_probabilities_holdout[,"fast_growth"]
RMSE(data_holdout$rf_p_prediction, data_holdout$fast_growth)

# ROC curve on holdout
roc_obj_holdout <- roc(data_holdout$fast_growth, data_holdout[, "rf_p_prediction", drop=TRUE], quiet=TRUE)


# AUC
as.numeric(roc_obj_holdout$auc)

# Get expected loss on holdout with optimal threshold
holdout_threshold <- coords(roc_obj_holdout, x = best_threshold[["rf_p"]] , input= "threshold",
                           ret="all", transpose = FALSE)
expected_loss_holdout <- (holdout_threshold$fp*FP + holdout_threshold$fn*FN)/length(data_holdout$fast_growth)
expected_loss_holdout


# Classification forest, Split by Gini, majority vote in each tree, majority vote over trees
# Show expected loss with classification RF and fast_growth majority voting to compare

train_control <- trainControl(
  method = "cv",
  n = 5
)
train_control$verboseIter <- TRUE

set.seed(2022)
rf_model_f <- train(
  formula(paste0("fast_growth_f ~ ", paste0(rfvars , collapse = " + "))),
  method = "ranger",
  data = data_train,
  tuneGrid = tune_grid,
  trControl = train_control
)

# Predict on both samples
data_train$rf_f_prediction_class <-  predict(rf_model_f,type = "raw")
data_holdout$rf_f_prediction_class <- predict(rf_model_f, newdata = data_holdout, type = "raw")

#We use predicted classes to calculate expected loss based on our loss fn
fp <- sum(data_holdout$rf_f_prediction_class == "fast_growth" & data_holdout$fast_growth_f == "no_fast_growth")
fn <- sum(data_holdout$rf_f_prediction_class == "no_fast_growth" & data_holdout$fast_growth_f == "fast_growth")
(fp*FP + fn*FN)/length(data_holdout$fast_growth)

# Summary results###############################################################

nvars[["rf_p"]] <- length(rfvars)

summary_results <- data.frame("Number of predictors" = unlist(nvars),
                              "CV RMSE" = unlist(CV_RMSE),
                              "CV AUC" = unlist(CV_AUC),
                              "CV threshold" = unlist(best_tresholds),
                              "CV expected Loss" = unlist(expected_loss))

model_names <- c("Logit X1", "Logit X4",
                 "Logit LASSO","RF probability")
summary_results <- summary_results %>%
  filter(rownames(.) %in% c("X1", "X4", "LASSO", "rf_p"))
rownames(summary_results) <- model_names

summary_results
















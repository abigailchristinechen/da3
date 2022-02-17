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


#Prob. LOGIT models #################################################################
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

#Apply different thresholds:
# 0.5 same as before
thresholds <-
  ifelse(data_holdout$best_logit_no_loss_pred < 0.5, "no_default", "default") %>%
  factor(levels = c("no_default", "default"))
cm_object1b <- confusionMatrix(thresholds,data_holdout$default_f)
cm1b <- cm_object1b$table
cm1b















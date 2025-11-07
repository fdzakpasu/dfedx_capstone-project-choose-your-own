
### HarvardX Capstone Project: Choose Your Own ###
##  Diabetes Risk Classification Using Machine Learning Algorithms: The Behavioural Risk Factor Surveillance System
##                      Francis Dzakpasu


## Installing and loading required packages and libraries
if(!require(rlang)) install.packages("rlang", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(Metrics)) install.packages("Metrics", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")

if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(R.utils)) install.packages("R.utils", repos = "http://cran.us.r-project.org")
if(!require(tictoc)) install.packages("tictoc", repos = "http://cran.us.r-project.org")
if(!require(tidymodels)) install.packages("tidymodels", repos = "http://cran.us.r-project.org")

if(!require(probably)) install.packages("probably", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
if(!require(vip)) install.packages("vip", repos = "http://cran.us.r-project.org")
if(!require(cluster)) install.packages("cluster", repos = "http://cran.us.r-project.org")

if(!require(purrr)) install.packages("purrr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(kknn)) install.packages("kknn", repos = "http://cran.us.r-project.org")

if(!require(MASS)) install.packages("MASS", repos = "http://cran.us.r-project.org")


## Some of the base packages required may need updating for the models to run. 
# update.packages()

library(tidyverse)
library(caret)
library(broom)
library(lubridate)
library(gam)
library(randomForest)
library(Rborist)
library(matrixStats)
library(rafalib)
library(naivebayes)

library(ggplot2)
library(ggthemes)
library(scales)
library(dslabs)
ds_theme_set()
library(knitr)
library(kableExtra)
library(recosystem)
library(R.utils)
library(tictoc)

library(dplyr)
library(readr)
library(tidymodels)
library(probably)
library(rpart.plot)
library(ranger)
library(vip)
library(cluster)
library(purrr)
tidymodels_prefer()

library(glmnet)
library(kknn)
library(MASS)
    

## Create and specify new working directory
# Specify the path for the new directory
new_dir_path <- "./my_CYOproj_folder"

# Check if the directory exists and create it if it doesn't
if (!dir.exists(new_dir_path)) {
  dir.create(new_dir_path)
  message("Directory '", new_dir_path, "' successfully created.")
} else {
  message("Directory '", new_dir_path, "' already exists.")
}

#set new directory
setwd("./my_CYOproj_folder")
getwd()

###Downloading the CDC Diabetes Health Indicators Dataset
## https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
## https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?resource=download

# Note: Dataset is unable to automatically download from the source into R. Therefore, the dataset has been downloaded and uploaded onto GitHub to enable automatic download with the code.


db <- "archive_diabetes health indicators dataset.zip"


download.file("https://github.com/fdzakpasu/dfedx_capstone-project-choose-your-own/raw/refs/heads/main/archive_diabetes%20health%20indicators%20dataset.zip", db)


db <- unzip(db)
db_binary <- read.csv("./archive_diabetes health indicators dataset/diabetes_binary_health_indicators_BRFSS2015.csv")

db_split <- read.csv("./archive_diabetes health indicators dataset/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

rm(db)


diabetes_binary <- as.data.frame(db_binary)
glimpse(diabetes_binary)

diabetes_split <- as.data.frame(db_split)
glimpse(diabetes_split)

##### To make the computation simple, a few variables will be selected.

### Train dataset and test dataset and set categorical variables as factors and their levels

###Train dataset
diabetes_df <- diabetes_binary %>% select("Diabetes_binary", "HighBP", "HighChol", "BMI", "PhysActivity", 
                                             "Sex", "Age", "Smoker", "HvyAlcoholConsump", "Education") %>%
  mutate(Diabetes_binary = relevel(factor(Diabetes_binary), ref="0",),
         HighBP = relevel(factor(HighBP), ref="0"),
         HighChol = relevel(factor(HighChol), ref="0"),
         PhysActivity = relevel(factor(PhysActivity), ref="0"),
         Sex = relevel(factor(Sex), ref="0"),
         Age = relevel(factor(Age), ref="1"),
         Smoker = relevel(factor(Smoker), ref="0"),
         HvyAlcoholConsump = relevel(factor(HvyAlcoholConsump), ref="0"),
         Education = relevel(factor(Education), ref="1"))
str(diabetes_df)


### Validation dataset
diabetes_val <- diabetes_split %>%select("Diabetes_binary", "HighBP", "HighChol", "BMI", "PhysActivity", 
                                          "Sex", "Age","Smoker", "HvyAlcoholConsump", "Education") %>%
  mutate(Diabetes_binary = relevel(factor(Diabetes_binary), ref="0",),
         HighBP = relevel(factor(HighBP), ref="0"),
         HighChol = relevel(factor(HighChol), ref="0"),
         PhysActivity = relevel(factor(PhysActivity), ref="0"),
         Sex = relevel(factor(Sex), ref="0"),
         Age = relevel(factor(Age), ref="1"),
         Smoker = relevel(factor(Smoker), ref="0"),
         HvyAlcoholConsump = relevel(factor(HvyAlcoholConsump), ref="0"),
         Education = relevel(factor(Education), ref="1"))
str(diabetes_val)


### Dataset exploration


# Visual exploration of the distribution of the predictor variables (features) of diabetes_df dataset

## Bar graph of Age
ggplot(diabetes_df, aes(x = Age, fill=Diabetes_binary)) +
  geom_bar() +
  scale_fill_brewer(palette = "Set2") + 
  labs(y = "Count",
       x = "Age category",
       fill = "Diabetes risk", 
       title = "Age Category distribution") + 
  theme_classic()  +
  theme(plot.title = element_text(hjust = 0.5))

## Bar graph of High Blood Pressure
ggplot(diabetes_df) +
  geom_bar(aes(x = HighBP, fill = Diabetes_binary)) +
  scale_fill_brewer(palette = "Set2") + 
  labs(y = "Count", 
       x = "High blood pressure",
       fill = "Diabetes risk", 
       title = "High Blood Pressure Distribution") + 
  theme_classic()  +
  theme(plot.title = element_text(hjust = 0.5))

## Bar graph of High Cholesterol
ggplot(diabetes_df) +
  geom_bar(aes(x = HighChol, fill = Diabetes_binary)) +
  scale_fill_brewer(palette = "Set2") + 
  labs(y = "Count", 
       x = "High cholesterol",
       fill = "Diabetes risk", 
       title = "High Cholesterol Distribution") + 
  theme_classic()  +
  theme(plot.title = element_text(hjust = 0.5))

## Bar graph of Physical Activity
ggplot(diabetes_df) +
  geom_bar(aes(x = PhysActivity, fill = Diabetes_binary)) +
  scale_fill_brewer(palette = "Set2") + 
  labs(y = "Count", 
       x = "Physical activity",
       fill = "Diabetes risk", 
       title = "Physical Activity Level Distribution") + 
  theme_classic()  +
  theme(plot.title = element_text(hjust = 0.5))

## Bar graph of Sex
ggplot(diabetes_df) +
  geom_bar(aes(x = Sex, fill = Diabetes_binary)) +
  scale_fill_brewer(palette = "Set2") + 
  labs(y = "Count", 
       x = "Gender",
       fill = "Diabetes risk", 
       title = "Gender Distribution") + 
  theme_classic()  +
  theme(plot.title = element_text(hjust = 0.5))

## Box plot for body mass index (BMI)
ggplot(diabetes_df) +
  geom_boxplot(aes(x = Diabetes_binary, y = BMI)) +
  ggtitle("Boxplot of BMI Distribution") +
  xlab("Diabetes risk") +
  ylab("Body mass index (BMI)") +
  theme_bw() +
  theme(axis.title.x = element_text(vjust = 0), 
        axis.title.y = element_text(vjust = 2),
        plot.title = element_text(hjust=0.5))

## Bar graph of Smoker
ggplot(diabetes_df) +
  geom_bar(aes(x = Smoker, fill = Diabetes_binary)) +
  scale_fill_brewer(palette = "Set2") + 
  labs(y = "Count", 
       x = "Smoking status",
       fill = "Diabetes risk", 
       title = "Smoking Status Distribution") + 
  theme_classic()  +
  theme(plot.title = element_text(hjust = 0.5))

## Bar graph of Heavy Alcohol Consumption
ggplot(diabetes_df) +
  geom_bar(aes(x = HvyAlcoholConsump, fill = Diabetes_binary)) +
  scale_fill_brewer(palette = "Set2") + 
  labs(y = "Count", 
       x = "Heavy alcohol consumption",
       fill = "Diabetes risk", 
       title = "Heavy Alcohol Consumption Distribution") + 
  theme_classic()  +
  theme(plot.title = element_text(hjust = 0.5))

## Bar graph of Education
ggplot(diabetes_df) +
  geom_bar(aes(x = Education, fill = Diabetes_binary)) +
  scale_fill_brewer(palette = "Set2") + 
  labs(y = "Count", 
       x = "Education level",
       fill = "Diabetes risk", 
       title = "Education Level Distribution") + 
  theme_classic()  +
  theme(plot.title = element_text(hjust = 0.5))




##### Machine learning Algorithms 

## The datase partitioning
# Partition the Diabetes_binary dataset into training (85%) and testing (15%) sets
# To allow the trained models to learn effectively and avoid overfitting.
# The test set will be a representative sample for an unbiased evaluation of the models’ 
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
partition_index <- createDataPartition(y = diabetes_df$Diabetes_binary, 
                                       times = 1, p = 0.15, list = FALSE)
diabetes_train <- diabetes_df[-partition_index,]
diabetes_test <- diabetes_df[partition_index,]


## Check the balance of the partitioned sets
# Outcome variable - Diabetes_binary
diabetes_train %>% dplyr::count(Diabetes_binary)

diabetes_test %>% dplyr::count(Diabetes_binary) 



### Logistic regression

# set.seed for reproducibility of the results 
set.seed(123) 

# Fit LASSO Model

# Set up CV folds
lasso_cv <- vfold_cv(diabetes_train, v = 10) # number of folds chosen to minimise the computation time

# LASSO logistic regression model specification
lasso_spec <- logistic_reg() %>%
  set_engine("glmnet") %>%
  set_args(mixture = 1, penalty = tune()) %>% # a pure LASSO model, which uses only the L1 penalty.
  set_mode("classification")

# Recipe
lasso_rec <- recipe(Diabetes_binary ~ ., 
                    data = diabetes_train) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

# Workflow (Recipe + Model)
lasso_wf <- workflow() %>%
  add_model(lasso_spec) %>%
  add_recipe(lasso_rec)

# Tune model: specify grid of parameters and tune
lasso_penalty_grid <- grid_regular(penalty(range = c(-5,3)), # log10 scale, lambda seq 10^-5 to 10^3
                             levels = 100
                             )

conflicted::conflicts_prefer(yardstick::accuracy)

lasso_tune_output <- tune_grid(lasso_wf,
                         resamples = lasso_cv,
                         metrics = metric_set(roc_auc, accuracy),
                         grid = lasso_penalty_grid
                         )

### Selecting the best fitting LASSO model

autoplot(lasso_tune_output) + theme_classic()


# Select "best" penalty by one standard error
lasso_best_penalty <- select_by_one_std_err(lasso_tune_output, 
                                         metric = "roc_auc", desc(penalty)
                                         )

# Define workflow with "best" penalty value
lasso_fit_wf <- finalize_workflow(lasso_wf, lasso_best_penalty)

# Use final_wf to fit final model with "best" penalty value
lasso_fit_bestpenalty <- fit(lasso_fit_wf, data = diabetes_train)


## Variable Importance
# get the original glmnet output
glmnet_output <- lasso_fit_bestpenalty %>% 
  extract_fit_parsnip() %>% 
  pluck("fit") 

plot(glmnet_output, xvar = "lambda", 
     label = TRUE, 
     col = rainbow(20)
     )

lasso_fit_bestpenalty %>% 
  extract_fit_engine() %>% 
  vip(num_features = 30) + theme_classic() 

# Extract the numerical values of the variable importance
lasso_var_imp <- as.data.frame(lasso_fit_bestpenalty %>% 
  extract_fit_engine() %>%
  vip::vi())

knitr::kable(lasso_var_imp)


## Model Evaluation
 # Accuracy and ROC AUC measures


lasso_tune_output %>%
  collect_metrics() %>%
  filter(penalty == lasso_best_penalty %>% 
           pull(penalty))              # get standard error of the mean 
                                       # (variability across multiple samples)


### Predictions on the test dataset and plots
lasso_pred <- predict(lasso_fit_bestpenalty, 
                      new_data = diabetes_test) %>%
  bind_cols(diabetes_test)

lasso_pred_labels <- c("0" = "Obs 0", "1"  = "Obs 1") #Obs - observed

# Plot for Age
ptest_lasso <- ggplot(lasso_pred, 
                      aes(x = Age, fill = .pred_class)) + 
  geom_bar(color = "black") + 
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = lasso_pred_labels)) + 
  labs(y = "Count", 
       fill = "Predictions", 
       title = "Age") + 
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

# A plot for High blood pressure
ptest1_lasso <- ggplot(lasso_pred, 
                       aes(x = HighBP, fill = .pred_class)) +
  geom_bar(color = "black") +
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = lasso_pred_labels)) +
  labs(y = "Count", 
       fill = "LASSO Predictions", 
       title = "High Blood Pressure") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

# Combining the plots
gridExtra::grid.arrange(ptest_lasso, ptest1_lasso, ncol = 2, 
                        top = "LASSO Predictions - Test Dataset")


## Confusion matrix
confmat_lasso <- confusionMatrix(lasso_pred$.pred_class, 
                                 diabetes_test$Diabetes_binary)

accuracy_lasso <- confmat_lasso$overall["Accuracy"]
sensitivity_lasso <- confmat_lasso$byClass["Sensitivity"]
specificity_lasso <- confmat_lasso$byClass["Specificity"]
det_rate_lasso <- confmat_lasso$byClass["Detection Rate"]



models_evaluations <- tibble(Method = "Logistic regression (LASSO): test", 
                             Accuracy = accuracy_lasso,
                             Sensitivity = sensitivity_lasso,
                             Specificity = specificity_lasso,
                             Detection_Rate = det_rate_lasso) 

models_evaluations


### Predictions on the validation dataset - 50:50 split of diabetes
lasso_pred_val <- predict(lasso_fit_bestpenalty, new_data = diabetes_val) %>%
  bind_cols(diabetes_val)

lasso_pred_val_labels <- c("0" = "Obs 0", "1"  = "Obs 1") #Obs - observed

# Plot for Age
pval_lasso <- ggplot(lasso_pred_val, 
                     aes(x = Age, fill = .pred_class)) + 
  geom_bar(color = "black") + 
  scale_fill_brewer(palette = "Set3") +
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = lasso_pred_val_labels)) +
  labs(y = "Count", 
       fill = "LASSO Predictions", 
       title = "Age") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

# A plot for High blood pressure
pval1_lasso <- ggplot(lasso_pred_val, 
                      aes(x = HighBP, fill = .pred_class)) +
  geom_bar(color = "black") +
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = lasso_pred_val_labels)) +
  labs(y = "Count", 
       fill = "LASSO Predictions", 
       title = "High Blood Pressure") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

# Combining the plots
gridExtra::grid.arrange(pval_lasso, pval1_lasso, ncol = 2, 
                        top = "LASSO Predictions - Validation Dataset")

## Confusion matrix
confmat_lasso_val <- confusionMatrix(lasso_pred_val$.pred_class, 
                                 diabetes_val$Diabetes_binary)

accuracy_lasso_val <- confmat_lasso_val$overall["Accuracy"]
sensitivity_lasso_val <- confmat_lasso_val$byClass["Sensitivity"]
specificity_lasso_val <- confmat_lasso_val$byClass["Specificity"]
det_rate_lasso_val <- confmat_lasso_val$byClass["Detection Rate"]



models_evaluations <- bind_rows(models_evaluations,
                                tibble(Method = "Logistic regression (LASSO): validation", 
                                       Accuracy = accuracy_lasso_val,
                                       Sensitivity = sensitivity_lasso_val,
                                       Specificity = specificity_lasso_val,
                                       Detection_Rate = det_rate_lasso_val) 
                                ) 

knitr::kable(models_evaluations)


###################################################################################


# Fit Ridge Model

set.seed(123)


# Set up CV folds
ridge_cv <- vfold_cv(diabetes_train, v = 10) # number of folds chosen to minimise the computation time

# Ridge logistic regression model specification
ridge_spec <- logistic_reg() %>%
  set_engine("glmnet") %>%
  set_args(mixture = 0, penalty = tune()) %>% # This argument fixes the model to use only the L2 penalty, 
                                              # which is characteristic of ridge regression. 
  set_mode("classification")

# Recipe
ridge_rec <- recipe(Diabetes_binary ~ ., 
                    data = diabetes_train) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

# Workflow (Recipe + Model)
ridge_wf <- workflow() %>%
  add_model(ridge_spec) %>%
  add_recipe(ridge_rec)

# Tune model: specify grid of parameters and tune
ridge_penalty_grid <- grid_regular(penalty(range = c(-5,3)), 
                                   levels = 100
)

ridge_tune_output <- tune_grid(ridge_wf,
                               resamples = ridge_cv,
                               metrics = metric_set(roc_auc, accuracy),
                               grid = ridge_penalty_grid
)

### Selecting the best fitting LASSO model

autoplot(ridge_tune_output) + theme_classic()


# Select "best" penalty by one standard error
ridge_best_penalty <- select_by_one_std_err(ridge_tune_output, 
                                            metric = "roc_auc", desc(penalty)
)

# Define workflow with "best" penalty value
ridge_fit_wf <- finalize_workflow(ridge_wf, ridge_best_penalty)

# Use final_wf to fit final model with "best" penalty value
ridge_fit_bestpenalty <- fit(ridge_fit_wf, data = diabetes_train)


## Variable Importance
# get the original glmnet output
glmnet_output_ridge <- ridge_fit_bestpenalty %>% 
  extract_fit_parsnip() %>% 
  pluck("fit") 

plot(glmnet_output_ridge, xvar = "lambda", 
     label = TRUE, 
     col = rainbow(20)
)


ridge_fit_bestpenalty %>% 
  extract_fit_engine() %>% 
  vip(num_features = 30) + theme_classic() 

# Extract the numerical values of the variable importance
ridge_var_imp <- as.data.frame(ridge_fit_bestpenalty %>% 
  extract_fit_engine() %>%
  vip::vi())

knitr::kable(ridge_var_imp)


## Model Evaluation
# Accuracy and ROC AUC measures


ridge_tune_output %>%
  collect_metrics() %>%
  filter(penalty == ridge_best_penalty %>% 
           pull(penalty))              # get standard error of the mean (variability across multiple samples)


### Predictions on the test dataset and plots
ridge_pred <- predict(ridge_fit_bestpenalty, 
                      new_data = diabetes_test) %>%
  bind_cols(diabetes_test)

ridge_pred_labels <- c("0" = "Obs 0", "1"  = "Obs 1") #Obs - observed

# Plot for Age
ptest_ridge <- ggplot(ridge_pred, 
                      aes(x = Age, fill = .pred_class)) + 
  geom_bar(color = "black") +  
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = ridge_pred_labels)) + 
  labs(y = "Count", 
       fill = "Predictions", 
       title = "Age") + 
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

# A plot for High blood pressure
ptest1_ridge <- ggplot(ridge_pred, 
                       aes(x = HighBP, fill = .pred_class)) +
  geom_bar(color = "black") +
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = ridge_pred_labels)) +
  labs(y = "Count", 
       fill = "Predictions", 
       title = "High Blood Pressure") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

# Combining the plots
gridExtra::grid.arrange(ptest_ridge, ptest1_ridge, ncol = 2, 
                        top = "Ridge Predictions - Test Dataset")


## Confusion matrix
confmat_ridge <- confusionMatrix(ridge_pred$.pred_class, 
                                 diabetes_test$Diabetes_binary)

accuracy_ridge <- confmat_ridge$overall["Accuracy"]
sensitivity_ridge <- confmat_ridge$byClass["Sensitivity"]
specificity_ridge <- confmat_ridge$byClass["Specificity"]
det_rate_ridge <- confmat_ridge$byClass["Detection Rate"]



models_evaluations <- bind_rows(models_evaluations,
                                tibble(Method = "Logistic regression (Ridge): test", 
                                       Accuracy = accuracy_lasso,
                                       Sensitivity = sensitivity_ridge,
                                       Specificity = specificity_ridge,
                                       Detection_Rate = det_rate_ridge)) 

knitr::kable(models_evaluations)


### Predictions on the validation dataset - 50:50 split of diabetes
ridge_pred_val <- predict(ridge_fit_bestpenalty, new_data = diabetes_val) %>%
  bind_cols(diabetes_val)

ridge_pred_val_labels <- c("0" = "Obs 0", "1"  = "Obs 1") #Obs - observed

# Plot for Age
pval_ridge <- ggplot(ridge_pred_val, 
                     aes(x = Age, fill = .pred_class)) + 
  geom_bar(color = "black") + 
  scale_fill_brewer(palette = "Set3") +
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = ridge_pred_val_labels)) +
  labs(y = "Count", 
       fill = "Predictions", 
       title = "Age") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))


# A plot for High blood pressure
pval1_ridge <- ggplot(ridge_pred_val, 
                      aes(x = HighBP, fill = .pred_class)) +
  geom_bar(color = "black") +
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = ridge_pred_val_labels)) +
  labs(y = "Count", 
       fill = "Predictions", 
       title = "High Blood Pressure") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

# Combining the plots
gridExtra::grid.arrange(pval_ridge, pval1_ridge, ncol = 2, 
                        top = "Ridge Predictions - Validation Dataset")


## Confusion matrix
confmat_ridge_val <- confusionMatrix(ridge_pred_val$.pred_class, 
                                     diabetes_val$Diabetes_binary)

accuracy_ridge_val <- confmat_ridge_val$overall["Accuracy"]
sensitivity_ridge_val <- confmat_ridge_val$byClass["Sensitivity"]
specificity_ridge_val <- confmat_ridge_val$byClass["Specificity"]
det_rate_ridge_val <- confmat_ridge_val$byClass["Detection Rate"]



models_evaluations <- bind_rows(models_evaluations,
                                tibble(Method = "Logistic regression (Ridge): validation", 
                                       Accuracy = accuracy_ridge_val,
                                       Sensitivity = sensitivity_ridge_val,
                                       Specificity = specificity_ridge_val,
                                       Detection_Rate = det_rate_ridge_val) 
) 

knitr::kable(models_evaluations)



####################################################################################

#### K-Nearest Neighbors (KNN) ####
set.seed(123)

## Fit KNN Model
knn_spec <- nearest_neighbor() %>%           # General model type
  set_args(neighbors = tune()) %>%           # Model tuning parameter
  set_engine(engine = "kknn") %>%            # Engine name
  set_mode("classification")

knn_cv <- vfold_cv(diabetes_train, v = 5) # Supply dataset and number of folds
                                          # A small number of folds chosen to minimise the computation time

knn_rec <- recipe(Diabetes_binary ~ HighBP + BMI + HighChol + Age + 
                    Sex + PhysActivity + HvyAlcoholConsump + 
                    Smoker + Education, data = diabetes_train) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

knn_wf <- workflow() %>%
  add_model(knn_spec) %>%                  # Model specification object
  add_recipe(knn_rec)                 # Data preprocessing recipe object

knn_param_tuning_grid <- grid_regular(neighbors(range = c(1, 100)),   # The minimum and maximum values for the neighbors
                                  levels = 5                          # the number of neighbors values
                                  )

### There is a conflict between the recipe and stringr packages, so the stringr is detached and uploaded again later 

conflicted::conflicts_prefer(recipes::accuracy)

# detach("package:stringr", unload = TRUE)


knn_tune_output <- tune_grid(knn_wf,
                             resamples = knn_cv,
                             metrics = metric_set(roc_auc, accuracy),
                             grid = knn_param_tuning_grid
                             )                          # NOTE: THIS STAGE WILL TAKE SEVERAL HOURS TO COMPLETE                                


###Select best fitting number of neighbors
autoplot(knn_tune_output) + theme_classic()


## Choose neighbors value that leads to the highest neighbors within 1 standard error.
knn_tune_output %>% 
  select_by_one_std_err(metric = "roc_auc", 
                        desc(neighbors)
                        )                   # The desc(neighbors) sorts the data from highest to
                                            # lowest number of neighbors (most simple -> most complex)
knn_tune_output %>% 
  select_by_one_std_err(metric = "accuracy", 
                        desc(neighbors)
                        )
best_neighbors <- select_by_one_std_err(knn_tune_output, 
                                        metric = "roc_auc", 
                                        desc(neighbors)
                                        )
fit_wf_knn <- finalize_workflow(knn_wf, 
                                best_neighbors) 
fit_bestneighbors <- fit(fit_wf_knn, 
                         data = diabetes_train)

# Show evaluation metrics for different values of neighbors, ordered
knn_tune_output %>% show_best(metric = "roc_auc")

knn_tune_output %>% show_best(metric = "accuracy")


# library("stringr")


### Predictions and plots
# Applying the best model to make predictions on testing dataset
knn_pred <- predict(fit_bestneighbors, 
                    new_data = diabetes_test) %>%
  bind_cols(diabetes_test)

knn_pred_labels <- c("0" = "Obs 0", "1"  = "Obs 1") #Obs - observed

# A plot to illustrate the model performance using BMI as example
ptest_knn <- ggplot(knn_pred, 
                    aes(x = BMI, fill = .pred_class)) +
  geom_histogram(bins = 10, color = "black") + 
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = knn_pred_labels)) + 
  labs(y = "Count", 
       fill = "Predictions", 
       title = "BMI") + 
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

# A plot for High blood pressure
ptest1_knn <- ggplot(knn_pred, 
                     aes(x = HighBP, fill = .pred_class)) +
  geom_bar(color = "black") +
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = knn_pred_labels)) +
  labs(y = "Count", 
       fill = "Predictions", 
       title = "High Blood Pressure") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

# Combining the plots
gridExtra::grid.arrange(ptest_knn, ptest1_knn, ncol = 2, 
                        top = "KNN Predictions - Test Dataset")


## Confusion matrix
confmat_knn <- confusionMatrix(knn_pred$.pred_class, 
                                 diabetes_test$Diabetes_binary)

accuracy_knn <- confmat_knn$overall["Accuracy"]
sensitivity_knn <- confmat_knn$byClass["Sensitivity"]
specificity_knn <- confmat_knn$byClass["Specificity"]
det_rate_knn <- confmat_knn$byClass["Detection Rate"]



models_evaluations <- bind_rows(models_evaluations,
                                tibble(Method = "K-Nearest Neighbors (KNN): test", 
                                       Accuracy = accuracy_knn,
                                       Sensitivity = sensitivity_knn,
                                       Specificity = specificity_knn,
                                       Detection_Rate = det_rate_knn)
                                ) 

knitr::kable(models_evaluations)



# Predictions on validation dataset: 50:50 split of diabetes

knn_pred_val <- predict(fit_bestneighbors, 
                        new_data = diabetes_val) %>%
  bind_cols(diabetes_val)

knn_pred_val_labels <- c("0" = "Obs 0", "1"  = "Obs 1") #Obs - observed

# A plot to illustrate the model performance using BMI as example
pval_knn <- ggplot(knn_pred_val, 
                   aes(x = BMI, fill = .pred_class)) +
  geom_histogram(bins = 10, color = "black") + 
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = knn_pred_val_labels)) + 
  labs(y = "Count", 
       fill = "Predictions", 
       title = "BMI") + 
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))


# A plot for High blood pressure
pval1_knn <- ggplot(knn_pred_val, 
                    aes(x = HighBP, fill = .pred_class)) +
  geom_bar(color = "black") +
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = knn_pred_val_labels)) +
  labs(y = "Count", 
       fill = "Predictions", 
       title = "High Blood Pressure") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

# Combining the plots
gridExtra::grid.arrange(pval_knn, pval1_knn, ncol = 2, 
                        top = "KNN Predictions - Validation Dataset")


## Confusion matrix
confmat_knn_val <- confusionMatrix(knn_pred_val$.pred_class, 
                                     diabetes_val$Diabetes_binary)

accuracy_knn_val <- confmat_knn_val$overall["Accuracy"]
sensitivity_knn_val <- confmat_knn_val$byClass["Sensitivity"]
specificity_knn_val <- confmat_knn_val$byClass["Specificity"]
det_rate_knn_val <- confmat_knn_val$byClass["Detection Rate"]


models_evaluations <- bind_rows(models_evaluations,
                                tibble(Method = "K-Nearest Neighbors (KNN): validation", 
                                       Accuracy = accuracy_knn_val,
                                       Sensitivity = sensitivity_knn_val,
                                       Specificity = specificity_knn_val,
                                       Detection_Rate = det_rate_knn_val)
                                ) 

knitr::kable(models_evaluations)



############################################################################################

### Decision Tree ####

## Fit Decision Tree
set.seed(123)

dtree_spec <- decision_tree() %>%
  set_engine(engine = "rpart") %>%
  set_args(cost_complexity = tune(),
           min_n = 2,
           tree_depth = NULL) %>%
  set_mode("classification")


dtree_cv <- vfold_cv(diabetes_train, v = 6) # A small number of folds chosen to minimise computation time

dtree_rec <- recipe(Diabetes_binary ~ ., 
                   data = diabetes_train)

#Workfow
dtree_wf <- workflow() %>% 
  add_model(dtree_spec) %>%
  add_recipe(dtree_rec)

#parameter grid
dtree_param_grid <- grid_regular(cost_complexity(range = c(-5, -1)),
                           levels = 10)
#Model tuning
dtree_tune_res <- tune_grid(dtree_wf, 
                      resamples = dtree_cv, 
                      grid = dtree_param_grid, 
                      metrics = metric_set(accuracy, roc_auc)
                      )


## Select and Fit Best Tree
autoplot(dtree_tune_res) + theme_classic()


#the cost-complexity selection
best_complexity <- select_by_one_std_err(dtree_tune_res, 
                                         metric = 'roc_auc', 
                                         desc(cost_complexity))

#the final workflow
dtree_fit_wf <- finalize_workflow(dtree_wf, best_complexity)

#the final best fit decision tree model
fit_dtree_best <- fit(dtree_fit_wf, data = diabetes_train)


## Visualise the Tree

dtree_plpot <- fit_dtree_best %>%
  extract_fit_engine()

##Prune the tree
#Determine the optimal complexity parameter (cp)
printcp(dtree_plpot)
plotcp(dtree_plpot)


# Find the best cp value
best_cp <- dtree_plpot$cptable[which.min(dtree_plpot$cptable[, "xerror"]), "CP"]
best_cp
# Prune the tree
dtree_pruned <- rpart::prune(dtree_plpot, cp = best_cp)

#Plot
rpart.plot(dtree_pruned, type = 0, extra = 104, 
           fallen.leaves = TRUE, cex = 0.5, 
           box.palette="GnBu",
           branch.lty=3, shadow.col="gray", nn=TRUE, 
           roundint=FALSE)



## Variable Importance
# Variable importance metrics 
# Sum of the goodness of split measures (impurity reduction) for each split for which it was the primary variable.
vip_dtree <- fit_dtree_best %>%
  extract_fit_engine() %>%
  pluck('variable.importance')

vip_dtree <- as.data.frame(vip_dtree) 

names(vip_dtree) <- "Values"
knitr::kable(vip_dtree)


## Model Evaluation
dtree_tune_res %>% 
  select_by_one_std_err(metric = "accuracy", 
                        desc(cost_complexity))

dtree_tune_res %>% 
  select_by_one_std_err(metric = "roc_auc", 
                        desc(cost_complexity))


# Predictions on test dataset

dtree_pred <- predict(fit_dtree_best, 
                   new_data = diabetes_test) %>%
  bind_cols(diabetes_test)

dtree_pred_labels <- c("0" = "Obs 0", "1"  = "Obs 1") #Obs - observed


### A plot to illustrate the model performance using BMI as an example

ptest_dtree <- ggplot(dtree_pred, 
                      aes(x = BMI, fill = .pred_class)) +
  geom_histogram(bins = 10, color = "black") + 
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = dtree_pred_labels)) + 
  labs(y = "Count", 
       fill = "Predictions", 
       title = "BMI") + 
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

# A plot for High blood pressure
ptest1_dtree <- ggplot(dtree_pred, 
                       aes(x = HighBP, fill = .pred_class)) +
  geom_bar(color = "black") +
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = dtree_pred_labels)) +
  labs(y = "Count", 
       fill = "Predictions", 
       title = "High Blood Pressure") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

# Combining the plots
gridExtra::grid.arrange(ptest_dtree, ptest1_dtree, ncol = 2, 
                        top = "Decision Tree Predictions - Test Dataset")


## Confusion matrix
confmat_dtree <- confusionMatrix(dtree_pred$.pred_class, 
                               diabetes_test$Diabetes_binary)

accuracy_dtree <- confmat_dtree$overall["Accuracy"]
sensitivity_dtree <- confmat_dtree$byClass["Sensitivity"]
specificity_dtree <- confmat_dtree$byClass["Specificity"]
det_rate_dtree <- confmat_dtree$byClass["Detection Rate"]


models_evaluations <- bind_rows(models_evaluations,
                                tibble(Method = "Decision Tree: test", 
                                       Accuracy = accuracy_dtree,
                                       Sensitivity = sensitivity_dtree,
                                       Specificity = specificity_dtree,
                                       Detection_Rate = det_rate_dtree)
                                )

knitr::kable(models_evaluations)



# Predictions on validation dataset: 50:50 split of diabetes

dtree_pred_val <- predict(fit_dtree_best, 
                       new_data = diabetes_val) %>%
  bind_cols(diabetes_val)

dtree_pred_val_labels <- c("0" = "Obs 0", "1"  = "Obs 1") #Obs - observed

### A plot to illustrate the model performance using BMI as an example
pval_dtree <- ggplot(dtree_pred_val, 
                     aes(x = BMI, fill = .pred_class)) +
  geom_histogram(bins = 10, color = "black") + 
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = dtree_pred_val_labels)) + 
  labs(y = "Count", 
       fill = "Predictions", 
       title = "BMI") + 
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))


# A plot for High blood pressure
pval1_dtree <- ggplot(dtree_pred_val, 
                      aes(x = HighBP, fill = .pred_class)) +
  geom_bar(color = "black") +
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = dtree_pred_val_labels)) +
  labs(y = "Count", 
       fill = "Predictions", 
       title = "High Blood Pressure") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

# Combining the plots
gridExtra::grid.arrange(pval_dtree, pval1_dtree, ncol = 2, 
                        top = "Decision Tree Predictions - Validation Dataset")


## Confusion matrix
confmat_dtree_val <- confusionMatrix(dtree_pred_val$.pred_class, 
                                   diabetes_val$Diabetes_binary)

accuracy_dtree_val <- confmat_dtree_val$overall["Accuracy"]
sensitivity_dtree_val <- confmat_dtree_val$byClass["Sensitivity"]
specificity_dtree_val <- confmat_dtree_val$byClass["Specificity"]
det_rate_dtree_val <- confmat_dtree_val$byClass["Detection Rate"]


models_evaluations <- bind_rows(models_evaluations,
                                tibble(Method = "Decision Tree: validation", 
                                       Accuracy = accuracy_dtree_val,
                                       Sensitivity = sensitivity_dtree_val,
                                       Specificity = specificity_dtree_val,
                                       Detection_Rate = det_rate_dtree_val)
                                )

knitr::kable(models_evaluations)



################################################################################

#### Random Forests ######
 ## Fitting the Random Forest Model
set.seed(123)

# Model Specification
rf_spec <- rand_forest() %>%
  set_engine(engine = "ranger") %>% 
  set_args(
    mtry = NULL, # size of random subset of variables
    trees = 1000, # Number of trees
    min_n = 2,
    probability = FALSE, # FALSE: get hard predictions
    importance = "impurity"
  ) %>%
  set_mode("classification")

# Recipe
rf_rec <- recipe(Diabetes_binary ~ ., data = diabetes_train)

# Workflows
rf_wf <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(rf_rec)

# No tune_grid() or vfold_cv()
rf_fit <- fit(rf_wf, data = diabetes_train)


## Variable Importance
# Plot of the variable importance information
rf_fit %>% 
  extract_fit_engine() %>% 
  vip(num_features = 30) + theme_classic()

# Extract the numerical information on variable importance
rf_var_imp <- rf_fit %>% 
  extract_fit_engine() %>%
  vip::vi()

knitr::kable(as.data.frame(rf_var_imp))


## Model Evaluation
rf_fit


## Predictions on the testing dataset

rf_pred <- predict(rf_fit, 
                    new_data = diabetes_test) %>%
  bind_cols(diabetes_test)

rf_pred_labels <- c("0" = "Obs 0", "1"  = "Obs 1") #Obs - observed

# A plot to illustrate the model performance using BMI as example
ptest_rf <- ggplot(rf_pred, 
                   aes(x = BMI, fill = .pred_class)) +
  geom_histogram(bins = 10, color = "black") + 
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = rf_pred_labels)) + 
  labs(y = "Count", 
       fill = "Predictions", 
       title = "BMI") + 
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

# A plot for High blood pressure
ptest1_rf <- ggplot(rf_pred, 
                    aes(x = HighBP, fill = .pred_class)) +
  geom_bar(color = "black") +
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = rf_pred_labels)) +
  labs(y = "Count", 
       fill = "Predictions", 
       title = "High Blood Pressure") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

# Combining the plots
gridExtra::grid.arrange(ptest_rf, ptest1_rf, ncol = 2, 
                        top = "Random Forest Predictions - Test Dataset")


## Confusion matrix
confmat_rf <- confusionMatrix(rf_pred$.pred_class, 
                               diabetes_test$Diabetes_binary)

accuracy_rf <- confmat_rf$overall["Accuracy"]
sensitivity_rf <- confmat_rf$byClass["Sensitivity"]
specificity_rf <- confmat_rf$byClass["Specificity"]
det_rate_rf <- confmat_rf$byClass["Detection Rate"]



models_evaluations <- bind_rows(models_evaluations,
                                tibble(Method = "Random Forest: test", 
                                       Accuracy = accuracy_rf,
                                       Sensitivity = sensitivity_rf,
                                       Specificity = specificity_rf,
                                       Detection_Rate = det_rate_rf)
) 

knitr::kable(models_evaluations)



# Predictions on validation dataset: 50:50 split of diabetes

rf_pred_val <- predict(rf_fit, 
                        new_data = diabetes_val) %>%
  bind_cols(diabetes_val)

rf_pred_val_labels <- c("0" = "Obs 0", "1"  = "Obs 1") #Obs - observed

# A plot to illustrate the model performance using BMI as example
pval_rf <- ggplot(rf_pred_val, 
                  aes(x = BMI, fill = .pred_class)) +
  geom_histogram(bins = 10, color = "black") + 
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = rf_pred_val_labels)) + 
  labs(y = "Count", 
       fill = "Predictions", 
       title = "BMI") + 
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))



# A plot for High blood pressure
pval1_rf <- ggplot(rf_pred_val, 
                   aes(x = HighBP, fill = .pred_class)) +
  geom_bar(color = "black") +
  scale_fill_brewer(palette = "Set3") + 
  facet_wrap(~Diabetes_binary, 
             labeller = labeller(Diabetes_binary = rf_pred_val_labels)) +
  labs(y = "Count", 
       fill = "Predictions", 
       title = "High Blood Pressure") +
  theme_classic()  +
  theme(plot.title = element_text(hjust = 0.5))

# Combining the plots
gridExtra::grid.arrange(pval_rf, pval1_rf, ncol = 2, 
                        top = "Random Forest Predictions - Validation Dataset")


## Confusion matrix
confmat_rf_val <- confusionMatrix(rf_pred_val$.pred_class, 
                                   diabetes_val$Diabetes_binary)

accuracy_rf_val <- confmat_rf_val$overall["Accuracy"]
sensitivity_rf_val <- confmat_rf_val$byClass["Sensitivity"]
specificity_rf_val <- confmat_rf_val$byClass["Specificity"]
det_rate_rf_val <- confmat_rf_val$byClass["Detection Rate"]


models_evaluations <- bind_rows(models_evaluations,
                                tibble(Method = "Random Forest: validation", 
                                       Accuracy = accuracy_rf_val,
                                       Sensitivity = sensitivity_rf_val,
                                       Specificity = specificity_rf_val,
                                       Detection_Rate = det_rate_rf_val)
) 

knitr::kable(models_evaluations)


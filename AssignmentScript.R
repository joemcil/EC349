# ---
# Data Collection
# ---

library(jsonlite)

cat("\014")  
rm(list=ls())

setwd("/Users/mcilwaine/Desktop/Uni/Year 4/EC349/Data Science Project")

#Load JSON Data
business_data <- 
  stream_in(file("yelp_dataset/yelp_academic_dataset_business.json")) 
checkin_data  <- 
  stream_in(file("yelp_dataset/yelp_academic_dataset_checkin.json")) 
tip_data  <- stream_in(file("yelp_dataset/yelp_academic_dataset_tip.json")) 
#Load Small Datasets for Reviews and Users
load("yelp_dataset/yelp_review_small.Rda")
load("yelp_dataset/yelp_user_small.Rda")


# ---
# Data Understanding: Carried out in console & code not 
# included here for simplicity. Findings in write-up file.
# ---


# ---
# Data Preparation Part 1: Cleaning Variables
# ---

# To avoid confusion between "stars" meanings in datasets
colnames(business_data)[colnames(business_data) == "stars"] <- "business_stars"
colnames(review_data_small)[colnames(review_data_small) == "stars"] <- 
  "review_stars"

#Encoding
business_data$BusinessAcceptsCreditCards <- 
  as.numeric(business_data$attributes$BusinessAcceptsCreditCards == "True")

# Feature Engineering. NOTE: can take a few
# minutes to run, but should not take too long.

library(stringr)

# Create a new column to store the count of dates
checkin_data$checkin_count <- str_count(checkin_data$date, 
                                        "\\d{4}-\\d{2}-\\d{2}")

library(dplyr)
library(tidyr)
library(lubridate)

# Split the comma-separated dates into a list
checkin_data$date_list <- strsplit(checkin_data$date, ", ")

# Convert each date in the list to a date format
checkin_data$date_list <- lapply(checkin_data$date_list, function(dates) {
  as.POSIXct(dates, format="%Y-%m-%d %H:%M:%S")
})

# Extract the first 4 characters (year) from each list of dates
checkin_data$year <- sapply(checkin_data$date_list, function(dates) {
  if (length(dates) > 0) {
    str_extract(as.character(year(dates[1])), "^\\d{4}")
  } else {
    NA
  }
})

checkin_data$year <- as.integer(checkin_data$year)
checkin_data$opp_years <- 2023-checkin_data$year
checkin_data$checkins_per_year <- 
  checkin_data$checkin_count/checkin_data$opp_years


# ---
# Data Preparation Part 2: Merging Datasets
# ---

# Getting subsets of the datasets with only the features we want, for merge
business_merge <- subset(business_data, select = c(business_id, business_stars, 
                                                   is_open, 
                                                   BusinessAcceptsCreditCards))
review_merge <- subset(review_data_small, select = c(review_id, business_id, 
                                                     user_id, review_stars, 
                                                     text, useful, funny, 
                                                     cool))
user_merge <- subset(user_data_small, select = c(user_id, average_stars))
checkin_merge <- subset(checkin_data, select = c(business_id, 
                                                 checkins_per_year))


# Merging
merged_data <- merge(business_merge, review_merge, by = "business_id", 
                     all = TRUE)
merged_data <- merge(merged_data, user_merge, by = "user_id", all = TRUE)
merged_data <- merge(merged_data, checkin_merge, by = "business_id", 
                     all = TRUE)


# Check for number of rows with missing data post-merge
rows_with_missing_values <- rowSums(is.na(merged_data)) > 0
count_missing_rows <- sum(rows_with_missing_values)
count_missing_rows/nrow(merged_data) #returns 0.848


# Drop these rows
sample_merged_data <- na.omit(merged_data) # leaves us with 253,265 observations



# ---
# Data Preparation Part 3: NLP & Sentiment Analysis
# NOTE: Separated from earlier feature engineering
# code as this is more computationally expensive so allows
# subsetting of merged data if needed.
# ---

# Take sample if needed for CPU
set.seed(1)
sample_merged_data <- sample_merged_data[sample(nrow(sample_merged_data), 
                                                size = 5000), ]


# NLP For Review Text
library(tm)
corpus <- iconv(sample_merged_data$text)
corpus <- Corpus(VectorSource(corpus))

corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
cleanset <- tm_map(corpus, removeWords, stopwords('english'))
cleanset <- tm_map(cleanset, stemDocument)
cleanset <- tm_map(cleanset, stripWhitespace)

# Works better when running the code above beforehand, rather than all in one
library(sentimentr)
sentiment_scores <- lapply(cleanset$content, function(x) {
  sentiment(x)
})


sentiment_df <- do.call(rbind, sentiment_scores)
sample_merged_data$sentiment_score <- sentiment_df$sentiment

# Can now drop the text colummn as well as the IDs
sample_merged_data <- subset(sample_merged_data, select = 
                               -c(text, user_id, business_id, review_id))


# ---
# Modelling: XGBoost
# ---

# Load Libraries
library(caret)
library(xgboost)

# Set Seed
set.seed(1)

# Data Splitting
index <- createDataPartition(sample_merged_data$review_stars, p = 0.7, 
                             list = FALSE)
training_data <- sample_merged_data[index, ]
testing_data <- sample_merged_data[-index, ]

# Define Predictors
predictors <- c("business_stars", "average_stars", "sentiment_score", 
                "useful", "funny", "cool", "is_open", 
                "BusinessAcceptsCreditCards", "checkins_per_year")

# Create DMatrix
createDMatrix <- function(data, label) {
  xgb.DMatrix(data = as.matrix(data[, predictors, drop = FALSE]), label = label)
}

# Set Cross-validation parameters
cv_params <- list(
  nfold = 5,
  stratified = FALSE,
  shuffle = TRUE
)

# Set XGBoost parameters
xgb_params <- list(
  objective = "reg:squarederror",
  nrounds = 200,
  max_depth = 5,
  eta = 0.1
)

# Perform Cross-validation
cv_model <- xgb.cv(
  params = xgb_params,
  data = createDMatrix(training_data, training_data$review_stars),
  nfold = cv_params$nfold,
  stratified = cv_params$stratified,
  shuffle = cv_params$shuffle,
  nrounds = xgb_params$nrounds,
  early_stopping_rounds = 50,
  verbose = 1
)

best_iteration <- cv_model$best_iteration

# Retrain the model with the optimal number of rounds
final_xgb_model <- xgboost(
  params = xgb_params,
  data = createDMatrix(training_data, training_data$review_stars),
  nrounds = best_iteration
)

# Make predictions on the training data
predictions <- predict(final_xgb_model, as.matrix(training_data[, predictors, 
                                                                drop = FALSE]))

# Evaluate the model
mse <- mean((predictions - training_data$review_stars)^2)

# Testing Data
X_test <- createDMatrix(testing_data, testing_data$review_stars)

# Make predictions on the testing data
predictions_test <- predict(final_xgb_model, as.matrix(testing_data[, 
                                                                    predictors, 
                                                                    drop = 
                                                                      FALSE]))

# Evaluate the model on testing data
mse_test <- mean((predictions_test - testing_data$review_stars)^2)

# Extract feature importance
importance <- xgb.importance(feature_names = 
                               colnames(training_data[, predictors, 
                                                      drop = FALSE]), 
                             model = final_xgb_model)

# Print feature importance
print(importance)



# ---
# Code Appendix: all commented out so that this script can be run from start 
# to finish if needed without confusing output
# ---
# Appendix 1: LASSO Skeleton Code
# Load Libraries
# library(caret)
# library(glmnet)
# 
# # Set Seed
# set.seed(1)
# 
# # Data Splitting
# index <- createDataPartition(sample_merged_data$review_stars, p = 0.7, list = FALSE)
# training_data <- sample_merged_data[index, ]
# testing_data <- sample_merged_data[-index, ]
# 
# # Define Predictors
# predictors <- c("business_stars", "average_stars", "sentiment_score", "useful", "funny", "cool", "is_open", "BusinessAcceptsCreditCards", "checkins_per_year")
# 
# # Scale the features
# scaled_training_data <- scale(training_data[, predictors])
# scaled_testing_data <- scale(testing_data[, predictors])
# 
# # Define the response variable
# response_variable <- training_data$review_stars
# 
# # Train LASSO model
# lasso_model <- cv.glmnet(scaled_training_data, response_variable, alpha = 1)
# 
# # Make predictions on the training data
# predictions <- predict(lasso_model, newx = scaled_training_data, s = "lambda.min")
# 
# # Evaluate the model on training data
# mse <- mean((predictions - response_variable)^2)
# 
# # Testing Data
# # Make predictions on the testing data
# predictions_test <- predict(lasso_model, newx = scaled_testing_data, s = "lambda.min")
# 
# # Evaluate the model on testing data
# mse_test <- mean((predictions_test - testing_data$review_stars)^2)
# 
# 
# 
# # Appendix 2: Ridge Skeleton Code
# # Load Libraries
# library(caret)
# library(glmnet)
# 
# # Set Seed
# set.seed(1)
# 
# # Data Splitting
# index <- createDataPartition(sample_merged_data$review_stars, p = 0.7, list = FALSE)
# training_data <- sample_merged_data[index, ]
# testing_data <- sample_merged_data[-index, ]
# 
# # Define Predictors
# predictors <- c("business_stars", "average_stars", "sentiment_score")
# 
# # Scale the features
# scaled_training_data <- scale(training_data[, predictors])
# scaled_testing_data <- scale(testing_data[, predictors])
# 
# # Define the response variable
# response_variable <- training_data$review_stars
# 
# # Train Ridge model
# ridge_model <- cv.glmnet(scaled_training_data, response_variable, alpha = 0)
# 
# # Make predictions on the training data
# predictions_ridge <- predict(ridge_model, newx = scaled_training_data, s = "lambda.min")
# 
# # Evaluate the model on training data
# mse_ridge <- mean((predictions_ridge - response_variable)^2)
# 
# # Testing Data
# # Make predictions on the testing data
# predictions_test_ridge <- predict(ridge_model, newx = scaled_testing_data, s = "lambda.min")
# 
# # Evaluate the model on testing data
# mse_test_ridge <- mean((predictions_test_ridge - testing_data$review_stars)^2)
# 
# 
# 
# #Appendix 3: Bagging Skeleton Code
# # Load Libraries
# library(ipred)
# library(caret)
# 
# # Set Seed
# set.seed(1)
# 
# # Data Splitting
# index <- createDataPartition(sample_merged_data$review_stars, p = 0.7, list = FALSE)
# training_data <- sample_merged_data[index, ]
# testing_data <- sample_merged_data[-index, ]
# 
# # Define Predictors
# predictors <- c("business_stars", "average_stars", "sentiment_score", "useful", "funny", "cool", "is_open", "BusinessAcceptsCreditCards", "checkins_per_year")
# 
# # Set Bagging parameters
# bagging_params <- list(
#   nbagg = 100,  # Number of bootstrap samples
#   coob = TRUE   # Use out-of-bag samples for prediction
# )
# 
# # Create the Bagging model
# bagging_model <- bagging(
#   formula = review_stars ~ .,
#   data = training_data,
#   nbagg = bagging_params$nbagg,
#   coob = bagging_params$coob
# )
# 
# # Make predictions on the training data
# predictions <- predict(bagging_model, newdata = training_data)
# 
# # Evaluate the model
# mse <- mean((predictions - training_data$review_stars)^2)
# 
# # Make predictions on the testing data
# predictions_test <- predict(bagging_model, newdata = testing_data)
# 
# # Evaluate the model on testing data
# mse_test <- mean((predictions_test - testing_data$review_stars)^2)
# 
# 
# 
# #Appendix 4: Random Forest Skeleton Code
# # Load Libraries
# library(caret)
# library(randomForest)
# 
# # Set Seed
# set.seed(1)
# 
# # Data Splitting
# index <- createDataPartition(sample_merged_data$review_stars, p = 0.7, list = FALSE)
# training_data <- sample_merged_data[index, ]
# testing_data <- sample_merged_data[-index, ]
# 
# # Define Predictors
# predictors <- c("business_stars", "average_stars", "sentiment_score", "useful", "funny", "cool", "is_open", "BusinessAcceptsCreditCards", "checkins_per_year")
# 
# # Train Random Forest model
# rf_model <- randomForest(review_stars ~ ., data = training_data, ntree = 100)
# 
# # Make predictions on the training data
# predictions <- predict(rf_model, training_data)
# 
# # Evaluate the model
# mse <- mean((predictions - training_data$review_stars)^2)
# 
# # Testing Data
# # Make predictions on the testing data
# predictions_test <- predict(rf_model, testing_data)
# 
# # Evaluate the model on testing data
# mse_test <- mean((predictions_test - testing_data$review_stars)^2)



# END
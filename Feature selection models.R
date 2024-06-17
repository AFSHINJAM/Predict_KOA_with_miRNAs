library(readxl)
library(mixOmics)
library(glmnet)
library(dplyr)
library(caret)
library(ROCR)

#################################################
# data prep

# set working directory
setwd("~/XXXXX")

#load data
DATA <- read_excel("data/data.xlsx")

# Custom Control Parameters
DATA$`no-prog/prog` <- as.factor(DATA$`no-prog/prog`)
DATA$Gender <- as.factor(DATA$Gender)
DATA$RACE <- as.factor(DATA$RACE)

set.seed(222)
ind <- sample(2, nrow(DATA), replace = T, prob = c(0.8, 0.2))
train <- DATA[ind==1,]
test <- DATA[ind==2,]

# Custom Control Parameters
custom <- trainControl(method = "repeatedcv",
                       number = 5,
                       repeats = 5,
                       verboseIter = T)

################################################################################

##### ELastic Net model
set.seed(1234)
en <- train(`no-prog/prog` ~ .,
            train, preProcess= c("center", "scale"),
            method= 'glmnet', classWeights = c(1, 1),
            tuneGrid = expand.grid(alpha= seq(0,1 , length= 10),
                                   lambda = seq(0.0001,1, length=10)), 
            trControl= custom)

# Plot Results
plot(en)
plot(en$finalModel, xvar = 'lambda', label=T)
plot(en$finalModel, xvar = 'dev', label=T)
plot(varImp(en), 10)
plot(varImp(en, scale=T), 10)
p1 <- predict(en, test)
xtab <- table(p1, test$`no-prog/prog`)
confusionMatrix(xtab, positive = "1")

##Finding best threshold in ElasticNet model

library(ROCR)
pred <- predict(en, test, type='prob')[,2]
pred <- prediction(pred, test$`no-prog/prog`)
eval <- performance(pred, "acc")
plot(eval)
abline(h=0.7666667, v=0.6818002)

#Identify best values for cutoff in the binary outcome
max<- which.max(slot(eval, "y.values")[[1]])
max
slot(eval, "y.values")[[1]][max]
acc <- slot(eval, "y.values")[[1]][max]
cut<- slot(eval, "x.values")[[1]][max]
print(c(Accuracy=acc, Cutoff = cut))

# Plot ROC curve
roc <- performance(pred, "tpr", "fpr")
plot(roc, colorize= T, main= "ROC curve", 
     ylab= "Sensitivity", xlab = "1- Specificity")
abline(a=0, b=1)

# Area under curve (AUC)
auc <- performance( pred, "auc")
auc<- unlist(slot(auc, "y.values"))
text(0.2, 0.8, paste("AUC =", round(auc, 3)), cex = 1.2)

p1 <- predict(en, test, type = "prob")
threshold <- 0.61
p1 <- ifelse(p1[, "1"] >= threshold, 1, 0)
xtab <- table(p1, test$`no-prog/prog`)
confusionMatrix(xtab, positive = "1")


################################################################################
##### LASSO model
set.seed(1234)
lasso <- train(`no-prog/prog` ~ .,
               train, preProcess= c("center", "scale"),
               method= 'glmnet',
               tuneGrid = expand.grid(alpha= 1,
                                      lambda = seq(0.0001,1, length=10)), 
               trControl= custom)

# Plot Results
plot(lasso)
plot(lasso$finalModel, xvar = 'lambda', label=T)
plot(lasso$finalModel, xvar = 'dev', label=T)
plot(varImp(lasso, scale=TRUE), top = 10, main = "glmnet")

p1 <- predict(lasso, test)
xtab <- table(p1, test$`no-prog/prog`)
confusionMatrix(xtab, positive = "1")

##Finding best threshold in LASSO model

library(ROCR)
pred <- predict(lasso, test, type='prob')[,2]
pred <- prediction(pred, test$`no-prog/prog`)
eval <- performance(pred, "acc")
plot(eval)
abline(h=0.7666667, v=0.6143980)

#Identify best values for cutoff in the binary outcome
max<- which.max(slot(eval, "y.values")[[1]])
max
slot(eval, "y.values")[[1]][max]
acc <- slot(eval, "y.values")[[1]][max]
cut<- slot(eval, "x.values")[[1]][max]
print(c(Accuracy=acc, Cutoff = cut))

# Plot ROC curve
roc <- performance(pred, "tpr", "fpr")
plot(roc, colorize= T, main= "ROC curve", 
     ylab= "Sensitivity", xlab = "1- Specificity")
abline(a=0, b=1)

# Area under curve (AUC)
auc <- performance(pred, "auc")
auc<- unlist(slot(auc, "y.values"))
text(0.2, 0.8, paste("AUC =", round(auc, 3)), cex = 1.2)

p1 <- predict(lasso, test, type = "prob")
threshold <- 0.6143980 
p1 <- ifelse(p1[, "1"] >= threshold, 1, 0)
xtab <- table(p1, test$`no-prog/prog`)
confusionMatrix(xtab, positive = "1")

################################################################################

## Ridge regression model

# Set seed for reproducibility
set.seed(1234)

# Train the Ridge regression model
Ridge <- train(`no-prog/prog` ~ .,
               data = train,
               preProcess = c("center", "scale"),
               method = 'glmnet',
               tuneGrid = expand.grid(alpha = 0,
                                      lambda = seq(0.0001, 1, length = 10)), 
               trControl = custom)

# Plot Results
plot(Ridge)
plot(Ridge$finalModel, xvar = 'lambda', label = TRUE)
plot(Ridge$finalModel, xvar = 'dev', label = TRUE)
plot(varImp(Ridge, scale = TRUE), top = 10, main = "glmnet")

##Finding best threshold in ridge regression model

library(ROCR)
pred <- predict(Ridge, test, type='prob')[,2]
pred <- prediction(pred, test$`no-prog/prog`)
eval <- performance(pred, "acc")
plot(eval)
abline(h=0.7333333, v=0.6661631)

#Identify best values for cutoff in the binary outcome
max<- which.max(slot(eval, "y.values")[[1]])
max
slot(eval, "y.values")[[1]][max]
acc <- slot(eval, "y.values")[[1]][max]
cut<- slot(eval, "x.values")[[1]][max]
print(c(Accuracy=acc, Cutoff = cut))

# Plot ROC curve
roc <- performance(pred, "tpr", "fpr")
plot(roc, colorize= T, main= "ROC curve", 
     ylab= "Sensitivity", xlab = "1- Specificity")
abline(a=0, b=1)

# Area under curve (AUC)
auc <- performance(pred, "auc")
auc<- unlist(slot(auc, "y.values"))
text(0.2, 0.8, paste("AUC =", round(auc, 3)), cex = 1.2)

p1 <- predict(Ridge, test, type = "prob")
threshold <- 0.6661631 
p1 <- ifelse(p1[, "1"] >= threshold, 1, 0)
xtab <- table(p1, test$`no-prog/prog`)
confusionMatrix(xtab, positive = "1")


################################################################################
## RANDOM FOREST model

# Train the random forest model
set.seed(1234)
grid <- expand.grid(mtry = c(1:10))
RF <- train(`no-prog/prog` ~ ., 
            data = train, 
            preProcess = c("center", "scale"),
            method = 'rf', 
            tuneGrid = grid,
            trControl = custom)

# Make predictions on the test set
p1 <- predict(RF, test)
xtab <- table(p1, test$`no-prog/prog`)
conf_mat <- confusionMatrix(xtab, positive = "1")

# Print confusion matrix
print("Confusion Matrix:")
print(conf_mat$table)

# Calculate sensitivity and specificity
TP <- conf_mat$table[2, 2]  
TN <- conf_mat$table[1, 1]  
FP <- conf_mat$table[1, 2]  
FN <- conf_mat$table[2, 1]  

sensitivity <- TP / (TP + FN)  
specificity <- TN / (TN + FP)  

# Print sensitivity and specificity
print(paste("Sensitivity:", round(sensitivity, 2)))
print(paste("Specificity:", round(specificity, 2)))

# Make predictions on test set for AUC calculation
pred <- predict(RF, test, type = 'prob')[,2]
pred_obj <- prediction(pred, test$`no-prog/prog`)
perf <- performance(pred_obj, "tpr", "fpr")

# Compute AUC
auc <- performance(pred_obj, "auc")
auc <- unlist(slot(auc, "y.values"))

# Print AUC
print(paste("AUC:", round(auc, 2)))

# Plot ROC curve
plot(perf, colorize = TRUE, main = "ROC Curve", 
     ylab = "Sensitivity", xlab = "1 - Specificity")
abline(a = 0, b = 1)  # Add diagonal line

# Add text for AUC value on the plot
text(0.5, 0.5, paste("AUC =", round(auc, 2)), cex = 1.2)

# Compute accuracy
accuracy <- sum(diag(conf_mat$table)) / sum(conf_mat$table)

# Print accuracy
print(paste("Accuracy:", round(accuracy, 2)))

################################################################################

# GBM model

library(gbm)
set.seed(1234)
# Define the grid of hyperparameters to search over
grid <- expand.grid(interaction.depth = seq(1,9,by=1),
                    n.trees = c(25,50,100, 250, 500),n.minobsinnode = 9,
                    shrinkage = c(0.1, 0.05, 0.01,0.1))
gbm <- train(`no-prog/prog` ~ . , train, preProcess= c("center", "scale"),
             method= 'gbm',  tuneGrid = grid,
             trControl= custom)
plot(gbm)
gbmImp <- varImp(gbm, scale = TRUE)
gbmImp
plot(gbmImp, 10)
p1 <- predict(gbm, test)
xtab <- table(p1, test$`no-prog/prog`)
confusionMatrix(xtab, positive = "1")

# Make predictions on test set for AUC calculation
pred <- predict(gbm, test, type = 'prob')[,2]
pred_obj <- prediction(pred, test$`no-prog/prog`)
perf <- performance(pred_obj, "tpr", "fpr")

# Compute AUC
auc <- performance(pred_obj, "auc")
auc <- unlist(slot(auc, "y.values"))

# Print AUC
print(paste("AUC:", round(auc, 2)))

################################################################################

## DT model

# Read data
DATA <- read_excel("data/data.xlsx")

# Convert necessary columns to factors
DATA$`no-prog/prog` <- as.factor(DATA$`no-prog/prog`)
DATA$Gender <- as.factor(DATA$Gender)
DATA$RACE <- as.factor(DATA$RACE)

# Split data into training and test sets
set.seed(222)
ind <- sample(2, nrow(DATA), replace = TRUE, prob = c(0.8, 0.2))
train <- DATA[ind == 1,]
test <- DATA[ind == 2,]

# Custom Control Parameters
custom <- trainControl(method = "repeatedcv",
                       number = 5,
                       repeats = 5,
                       verboseIter = TRUE,
                       classProbs = TRUE,
                       summaryFunction = twoClassSummary)

rename_columns <- function(df) {
  colnames(df) <- make.names(colnames(df))
  return(df)
}

train <- rename_columns(train)
test <- rename_columns(test)

levels(train$no.prog.prog) <- make.names(levels(train$no.prog.prog))
levels(test$no.prog.prog) <- make.names(levels(test$no.prog.prog))

# Set seed for reproducibility
set.seed(1234)

# Define grid for hyperparameter tuning
grid <- expand.grid(cp = seq(0.01, 0.1, by = 0.01))

# Train the model using the training set
DT <- train(no.prog.prog ~ .,
            data = train,
            preProcess = c("center", "scale"),
            method = 'rpart',
            tuneGrid = grid,
            trControl = custom,
            metric = "ROC")

print(DT)
plot(DT)

# Predict on the test set
test_prob <- predict(DT, test, type = "prob")[, 2]

# Use ROCR to find the best threshold
pred <- prediction(test_prob, test$no.prog.prog)
eval <- performance(pred, "acc")
plot(eval)
max <- which.max(slot(eval, "y.values")[[1]])
acc <- slot(eval, "y.values")[[1]][max]
cut <- slot(eval, "x.values")[[1]][max]
print(c(Accuracy = acc, Cutoff = cut))

# Apply the best threshold
threshold <- cut
test_pred <- ifelse(test_prob >= threshold, "X1", "X0")
test_pred <- factor(test_pred, levels = c("X0", "X1"))

# Generate confusion matrix
conf_matrix <- confusionMatrix(test_pred, test$no.prog.prog)
print(conf_matrix)

# Calculate AUC
roc_curve <- roc(test$no.prog.prog, test_prob)
auc_value <- auc(roc_curve)

# Extract performance metrics
acc <- conf_matrix$overall["Accuracy"]
sen <- conf_matrix$byClass["Sensitivity"]
spe <- conf_matrix$byClass["Specificity"]

# Print performance metrics
cat("AUC:", auc_value, "\n")
cat("Accuracy (ACC):", acc, "\n")
cat("Sensitivity (SEN):", sen, "\n")
cat("Specificity (SPE):", spe, "\n")

# Plot ROC curve
roc <- performance(pred, "tpr", "fpr")
plot(roc, colorize=TRUE, main="ROC curve", ylab="Sensitivity", xlab="1 - Specificity")
abline(a=0, b=1)
text(0.2, 0.8, paste("AUC =", round(auc_value, 3)), cex = 1.2)

################################################################################

## Recursive Feature Elimination model

# Read data
DATA <- read_excel("data/data.xlsx")

# Convert necessary columns to factors
DATA$`no-prog/prog` <- as.factor(DATA$`no-prog/prog`)
DATA$Gender <- as.factor(DATA$Gender)
DATA$RACE <- as.factor(DATA$RACE)

# Define control for RFE
control <- rfeControl(functions = rfFuncs, 
                      method = "repeatedcv", 
                      repeats = 5, 
                      number = 5) 

# Features
x <- DATA %>%
  select(-`no-prog/prog`) %>%
  as.data.frame()

# Target variable
y <- DATA$`no-prog/prog`

# Training: 80%; Test: 20%
set.seed(2021)
inTrain <- createDataPartition(y, p = .80, list = FALSE)[,1]

x_train <- x[inTrain, ]
x_test  <- x[-inTrain, ]

y_train <- y[inTrain]
y_train <- as.factor(y_train)
y_test  <- y[-inTrain]
y_test <- as.factor(y_test)

# Run RFE with ROC as the metric
result_rfe1 <- rfe(x = x_train, 
                  y = y_train, 
                  sizes = c(1:10),
                  rfeControl = control, 
                  metric = "ROC")
print(result_rfe1)

# Run RFE with accuracy as the metric
result_rfe2 <- rfe(x = x_train, 
                  y = y_train, 
                  sizes = c(1:10),
                  rfeControl = control, 
                  metric = "Accuracy")
print(result_rfe2)
predictors(result_rfe2)
ggplot(data = result_rfe2, metric = "Accuracy") + theme_bw()
################################################################################

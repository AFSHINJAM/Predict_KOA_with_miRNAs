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
DATA <- read_excel("C:/Users/afshi/OneDrive - McGill University/New folder/merged_data.xlsx")

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

##Finding best threshold in LASSO model

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
DATA <- read_excel("C:/Users/afshi/OneDrive - McGill University/New folder/merged_data.xlsx")

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

# Rename columns to make them valid R variable names
rename_columns <- function(df) {
  colnames(df) <- make.names(colnames(df))
  return(df)
}

train <- rename_columns(train)
test <- rename_columns(test)

# Ensure target variable levels are valid R variable names
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

# Print model summary
print(DT)

# Plot Results
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
DATA <- read_excel("C:/Users/afshi/OneDrive - McGill University/New folder/merged_data.xlsx")

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


# Print the selected features
predictors(result_rfe2)

# Print the results visually
ggplot(data = result_rfe2, metric = "Accuracy") + theme_bw()

################################################################################

## Sparse Partial Least Squares Regression

DATA <- read_excel("C:/Users/afshi/OneDrive - McGill University/New folder/merged_data.xlsx")

set.seed(123)
X<- DATA[,1:111]
Y<- DATA$`no-prog/prog`
srbct.splsda <- mixOmics:: splsda(X, Y, ncomp = 10, scale = TRUE)

plotIndiv(srbct.splsda , comp = 1:2, 
          group = DATA$`no-prog/prog`, ind.names = FALSE, 
          ellipse = TRUE, 
          legend = TRUE,legend.title = "Progressor", title = '(a) sPLSDA with confidence ellipses')

background = background.predict(srbct.splsda, comp.predicted=2, dist = "max.dist")

plotIndiv(srbct.splsda, comp = 1:2,
          group = DATA$`no-prog/prog`, ind.names = FALSE, 
          background = background, 
          legend = TRUE,legend.title = "Progressor", title = " (b) sPLSDA with prediction background")

perf.splsda.srbct <- perf(srbct.splsda, validation = "Mfold", scale=TRUE,
                          folds = 5, nrepeat = 50, 
                          progressBar = TRUE, auc = TRUE) 

# plot the outcome of performance evaluation across all ten components
plot(perf.splsda.srbct, col = color.mixo(5:7), sd = TRUE,
     legend.position = "horizontal")
perf.splsda.srbct$choice.ncomp

list.keepX <- c(1:20)
tune.splsda.srbct <- mixOmics:: tune.splsda(X, Y, ncomp = 2, 
                                            validation = 'Mfold',scale = TRUE,
                                            folds = 3, nrepeat = 100, 
                                            dist = 'max.dist', 
                                            measure = "AUC", 
                                            test.keepX = list.keepX,
                                            cpus = 2) 
plot(tune.splsda.srbct, col = color.jet(2))


tune.splsda.srbct$choice.ncomp$ncomp
tune.splsda.srbct$choice.keepX 
optimal.ncomp <- tune.splsda.srbct$choice.ncomp$ncomp
optimal.keepX <- tune.splsda.srbct$choice.keepX[1:optimal.ncomp]

final.splsda <- mixOmics:: splsda(X, Y, scale=TRUE,
                                  ncomp = 2, 
                                  keepX = c(2,1))

plotIndiv(final.splsda, comp = c(1,2), 
          group = DATA$`no-prog/prog`, ind.names = FALSE, 
          ellipse = TRUE, legend = TRUE,legend.title = "Progressor", 
          title = 'sPLS-DA, comp 1 & 2')

Y <- as.factor(Y)
legend=list(legend = levels(Y), 
            col = unique(color.mixo(Y)),
            title = "Progressor", 
            cex = 0.7)
cim <- cim(final.splsda, row.sideColors = color.mixo(Y), cluster="both",
           legend = legend)
perf.splsda.srbct <- perf(final.splsda, scale=TRUE,
                          folds = 5, nrepeat = 100, 
                          validation = "Mfold", dist = "max.dist",  
                          progressBar = FALSE)
par(mfrow=c(1,2))
plot(perf.splsda.srbct$features$stable[[1]], type = 'h', 
     ylab = 'Stability', lwd = 2,
     xlab = 'Features', cex.axis= 0.5,cex.lab=1,
     main = '(a) Comp 1', las =2)
plot(perf.splsda.srbct$features$stable[[2]], type = 'h', 
     ylab = 'Stability', 
     xlab = 'Features', cex.axis= 0.5,cex.lab=1,
     main = '(b) Comp 2', las =2)

plotVar(final.splsda, comp = c(1,2), cex = 4)


set.seed(123)
n <- nrow(DATA)
n_train <- round(0.8 * n)

# Randomly select indices for training and testing sets
train_indices <- sample(1:n, n_train, replace = FALSE)
test_indices <- setdiff(1:n, train_indices)

# Define X.train, Y.train, X.test, and Y.test
X.train <- scale(DATA[train_indices, 1:111], center = TRUE)
Y.train <- DATA$`no-prog/prog`[train_indices]
X.test <- scale(DATA[test_indices, 1:111], center = TRUE)
Y.test <- DATA$`no-prog/prog`[test_indices]
Y.test<- as.factor(Y.test)


train.splsda.srbct <- mixOmics:: splsda(X.train, Y.train, scale=TRUE,
                                        ncomp = 2, keepX = c(4,1))
predict.splsda.srbct <- predict(train.splsda.srbct, X.test, 
                                dist = "mahalanobis.dist")
predict.comp <- predict.splsda.srbct$class$mahalanobis.dist[,1]
table(predict.comp, Y.test)

# Confusion matrix
conf_matrix <- table(predict.comp, Y.test)
print(conf_matrix)

# Define the elements of the confusion matrix
TP <- conf_matrix[2, 2]
TN <- conf_matrix[1, 1]
FP <- conf_matrix[2, 1]
FN <- conf_matrix[1, 2]

auc.splsda = auroc(train.splsda.srbct, roc.comp = 1, print = FALSE)

auc.splsda = auroc(train.splsda.srbct, roc.comp = 2, print = FALSE)

# Calculate Sensitivity
sensitivity <- TP / (TP + FN)
print(paste("Sensitivity:", sensitivity))

# Calculate Specificity
specificity <- TN / (TN + FP)
print(paste("Specificity:", specificity))

################################################################################
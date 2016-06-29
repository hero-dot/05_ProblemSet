# Required libraries 
library(caret)
library(dplyr)
library(doParallel)
# install.packages("doParallel")
library(e1071)
# install.packages("e1071")

# Import the Data 
creditDefaultData = read.csv("default of credit card clients.csv",sep = ";")

# a. 
# Preprocessing of the data
creditDefaultData$SEX = as.factor(creditDefaultData$SEX)
creditDefaultData$EDUCATION = as.factor(creditDefaultData$EDUCATION)
creditDefaultData$MARRIAGE = as.factor(creditDefaultData$MARRIAGE)
creditDefaultData$default.payment.next.month = as.factor(creditDefaultData$default.payment.next.month)

# Remove unneeded columns
creditDefaultData = select(creditDefaultData,-ID)

# Use a 25:5 split to train the model
trainIndex <- createDataPartition(creditDefaultData$default.payment.next.month, p = .83, list = FALSE)
training <- creditDefaultData[trainIndex,] #training data (83% of observations)
testing  <- creditDefaultData[-trainIndex,] #test data (17% of observations)

#Create X Matrix
options(na.action='na.pass')
trainingMatrix <- model.matrix(default.payment.next.month ~ . - 1, data = training)
testMatrix <- model.matrix(default.payment.next.month ~ . - 1, data = testing)

# Add a train control 
tc <- trainControl(
  method = "repeatedcv",
  number = 3,
  repeats = 3)

# Train a kNN
cl <- makeCluster(detectCores())
registerDoParallel(cl)

caret.kNN <-  train(trainingMatrix,
    training$default.payment.next.month,
    preProcess = c("center", "scale"),
    method = "knn",
    metric = "Accuracy",
    tuneLength = 20,
    trControl = tc)

stopCluster(cl)
caret.kNN$finalModel
caret.kNN$results
summary(caret.kNN)
ggplot(caret.kNN)
prediction = predict(caret.kNN,testMatrix)
confusionMatrix(prediction,testing$default.payment.next.month)

# Train a logisitic regression
cl <- makeCluster(detectCores())
registerDoParallel(cl)

caret.logReg <- train(trainingMatrix,
                      training$default.payment.next.month,
                      preProcess=c("knnImpute"),
                      method = "glm",
                      family = "binomial",
                      metric = "Accuracy",
                      tuneLength = 1, #no tuning for log reg
                      trControl = tc)
stopCluster(cl)

prediction = predict(caret.logReg,testMatrix)
confusionMatrix(prediction,testing$default.payment.next.month)
plot(caret.logReg$finalModel)


# Train a decision tree
cl <- makeCluster(detectCores())
registerDoParallel(cl)

caret.ctree <- train(trainingMatrix,
                     training$default.payment.next.month,
                     preProcess=c("knnImpute"),
                     method = "ctree",
                     metric = "Accuracy",
                     tuneLength = 10,
                     trControl = tc)
stopCluster(cl)

plot(caret.ctree$finalModel)
prediction = predict(caret.ctree,testMatrix)
confusionMatrix(prediction,testing$default.payment.next.month)
plot(caret.ctree)

# Train a random forest
cl <- makeCluster(detectCores())
registerDoParallel(cl)

caret.rf <- train(trainingMatrix,
                  training$default.payment.next.month,
                  preProcess=c("knnImpute"),
                  method = "rf",
                  metric = "Accuracy",
                  tuneLength = 10,
                  trControl = tc)
stopCluster(cl)

prediction = predict(caret.rf,testMatrix)
confusionMatrix(prediction,testing$Survived)
plot(caret.rf)


# Provide a table akin to table 1

# b. 
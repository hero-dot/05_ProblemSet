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
creditDefaultData$PAY_0 = as.factor(creditDefaultData$PAY_0)
creditDefaultData$PAY_2 = as.factor(creditDefaultData$PAY_2)
creditDefaultData$PAY_3 = as.factor(creditDefaultData$PAY_3)
creditDefaultData$PAY_4 = as.factor(creditDefaultData$PAY_4)
creditDefaultData$PAY_5 = as.factor(creditDefaultData$PAY_5)
creditDefaultData$PAY_6 = as.factor(creditDefaultData$PAY_6)

# Use a 25:5 split to train the model
creditDefaultData = select(creditDefaultData,-ID)
trainIndex <- createDataPartition(creditDefaultData$default.payment.next.month, p = .83, list = FALSE)
training <- creditDefaultData[trainIndex,] #training data (83% of observations)
testing  <- creditDefaultData[-trainIndex,] #test data (17% of observations)

#Create X Matrix
options(na.action='na.pass')
trainMatrix <- model.matrix(default.payment.next.month ~ . - 1, data = training)
testMatrix <- model.matrix(default.payment.next.month ~ . - 1, data = testing)

# Add a train control 
tc <- trainControl(
  method = "repeatedcv",
  number = 3,
  repeats = 3)

# Train a kNN
cl <- makeCluster(detectCores())
registerDoParallel(cl)
caret.kNN <-  train(trainMatrix,
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
caret.logReg <- train(trainMatrix,
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
caret.ctree <- train(trainMatrix,
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
# caret.knn and caret.rf is missing

models = list(caret.logReg,caret.ctree)

ModName = NULL
ErrRaTr = NULL
ErrRaVal = NULL

for (mod in models) 
{
  ModName =rbind(ModName,mod$method) 
  ErrorTrain = 1 - getTrainPerf(mod)[[1]]
  ErrRaTr = rbind(ErrRaTr,ErrorTrain)
  
  pred = predict(mod,testMatrix)
  acc = confusionMatrix(pred,testing$default.payment.next.month)
  ErrorVal  = 1 - acc$overall[[1]]
  ErrRaVal = rbind(ErrRaVal,ErrorVal)
}

AllMetrics = data.frame(ModName,ErrRaTr,ErrRaVal)
# Area Ratio

# Area below Baseline = 0.5 

# Area below Best possible Curve
# 0.5*positives+(total-positives)*positives 

# AUC 
evalResults <- data.frame(Class = testing$default.payment.next.month)
evalResults$logreg <-predict(caret.logReg, testMatrix, type = "prob")[,"1"]
evalResults$ctree <- predict(caret.ctree, testMatrix, type = "prob")[,"1"]

evalResults$rf <- predict(caret.rf, testMatrix, type = "prob")[,"1"]
evalResults$gbm <- predict(caret.kNN, testMatrix, type = "prob")[,"1"]

liftData <- lift(Class ~ logreg + ctree, data = evalResults)

require(dplyr)

evalResults <- data.frame(Class = testing$y)
evalResults$lr <- lrpredict_prob[,"yes"]

evalResults %>%
  arrange(-logreg) %>%
  mutate(count=1:n()) %>%
  mutate(cumulativeHits = cumsum(.$Class=="1")) %>%
  mutate(cumulativePercentage = cumulativeHits/153,
         count = count / 1400)-> x
plot(x$count,x$cumulativePercentage,type="l",col="red",lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")



# b. 

# Explain the Sorting Smoothing Method

# Develop the graphs and apply to your classifiers

# How would you assess the performance of you random forest 
# vis a vis the methods in the paper

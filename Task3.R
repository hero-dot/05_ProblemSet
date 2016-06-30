library(caret)
library(doParallel)
# install.packages("pROC")
library(pROC)
library(ROCR)
# install.packages("ROCR")
library(dplyr)

bankData <- read.csv("bank.csv",sep = ";")
summary(bankData)

# a. 
# Logistic Regression 
trainIndex <- createDataPartition(bankData$y, p = .75, list = FALSE)
training <- bankData[trainIndex,] #training data (75% of observations)
testing  <- bankData[-trainIndex,] #test data (25% of observations)

logReg = glm(y~.,family = "binomial", data= training)

# Plot ROC for logReg
logReg.predict<-predict(logReg,testing,type= "response")
pred<-prediction(logReg.predict,testing$y)
perf <- performance(pred,"tpr","fpr")
plot(perf)

# Create X Matrix
options(na.action='na.pass')
trainingMatrix <- model.matrix(y ~ . - 1, data = training)
testMatrix <- model.matrix(y ~ . - 1, data = testing)

tc <- trainControl(
  method = "repeatedcv",
  number = 3,
  repeats = 3,
  summaryFunction = twoClassSummary,
  classProbs = T)

cl <- makeCluster(detectCores())
registerDoParallel(cl)

caret.logReg <- train(trainingMatrix,
                      training$y,
                      preProcess=c("knnImpute"),
                      method = "glm",
                      family = "binomial",
                      metric = "ROC",
                      tuneLength = 1,
                      trControl = tc)
stopCluster(cl)

# Confusion Matrix for the logReg
logReg.predict <- predict(caret.logReg, testMatrix)
confusionMatrix(logReg.predict,testing$y)

# Plot for the ROC 
logReg.predict <- predict(caret.logReg,testing,type="prob")
logRegROC<-roc(testing$y,logReg.predict[,"yes"],levels=rev(testing$y))
plot(logRegROC,type="S",print.thres=.5)

# Random Forest 
cl <- makeCluster(detectCores())
registerDoParallel(cl)

caret.rf <- train(trainingMatrix,
                  training$y,
                  preProcess=c("knnImpute"),
                  method = "rf",
                  metric = "ROC",
                  tuneLength = 10,
                  trControl = tc)
stopCluster(cl)

# Confusion matrix and ROC plot 
prediction = predict(caret.rf,testMatrix)
confusionMatrix(prediction,testing$y)
plot(caret.rf)

# Plot of sensitivity and sensibility
rf.predict = predict(caret.rf,testMatrix,type="prob")
rfROC<-roc(testing$y,rf.predict[,"yes"],levels=rev(testing$y))
plot(rfROC,type="S",print.thres=.5)

# Extract the ROC
getTrainPerf(caret.logReg)
getTrainPerf(caret.rf)

# b.

#Lift Diagrams
evalResults <- data.frame(Class = testing$y)
evalResults$logreg <- predict(caret.logReg, testMatrix, type = "prob")[,"yes"]
evalResults$rf <- predict(caret.rf, testMatrix, type = "prob")[,"yes"]

liftData <- lift(Class ~ rf +logreg, data = evalResults)

trellis.par.set(caretTheme())

liftplot = plot(liftData, lwd=3, values = 80, auto.key = list(columns = 3,
                                                              lines = TRUE,
                                                              lwd=3,
                                                              points = FALSE))

update(liftplot, par.settings = list(fontsize = list(text = 24)))
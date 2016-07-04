library(caret)
library(doParallel)
library(ROCR)
# install.packages("ROCR")

bankData <- read.csv("bank.csv",sep = ";")
summary(bankData)

# a. 
# Logistic Regression 
trainIndex <- createDataPartition(bankData$y, p = 0.66, list = FALSE)
training <- bankData[trainIndex,] #training data (75% of observations)
testing  <- bankData[-trainIndex,] #test data (25% of observations)

# Create X Matrix
options(na.action='na.pass')
trainMatrix <- model.matrix(y ~ . - 1, data = training)
testMatrix <- model.matrix(y ~ . - 1, data = testing)

tc <- trainControl(
  method = "repeatedcv",
  number = 3,
  repeats = 3,
  summaryFunction = twoClassSummary,
  classProbs = T)

# Logistic Regression
cl <- makeCluster(detectCores())
registerDoParallel(cl)
caret.logReg <- train(trainMatrix,
                      training$y,
                      preProcess=c("knnImpute"),
                      method = "glm",
                      family = "binomial",
                      metric = "Accuracy",
                      tuneLength = 1,
                      trControl = tc)
stopCluster(cl)

# Random Forest 
cl <- makeCluster(detectCores())
registerDoParallel(cl)

caret.rf <- train(trainMatrix,
                  training$y,
                  preProcess=c("knnImpute"),
                  method = "rf",
                  metric = "ROC",
                  tuneLength = 10,
                  trControl = tc)
stopCluster(cl)

# Model Evaluation 
# Plot ROC for logReg
logReg.predict<-predict(caret.logReg,testMatrix,type= "prob")[,"yes"]
predLogReg<-prediction(logReg.predict,testing$y)
perfLogReg <- performance(predLogReg,"tpr","fpr")
plot(perfLogReg, main = "ROC curves")
lines(c(-1,1),c(-1,1),lty = 2,add=T)

# Plot ROC for randomForest
rf.predict<-predict(caret.rf$finalModel,testMatrix,type= "prob")[,"yes"]
predRf<-prediction(rf.predict,testing$y)
perfRf <- performance(predRf,"tpr","fpr")
plot(perfRf,col = "red", add=T)

# Confusion Matrix for logReg
logReg.predict <- predict(caret.logReg, testMatrix)
confusionMatrix(logReg.predict,testing$y)

# Confusion matrix for randomForest 
prediction = predict(caret.rf,testMatrix)
confusionMatrix(prediction,testing$y)
plot(caret.rf)

# Extract the ROC
getTrainPerf(caret.logReg)
getTrainPerf(caret.rf)

# b.
#Lift Diagrams
evalResults <- data.frame(Class = testing$y)
evalResults$logreg <- predict(caret.logReg, testMatrix, type = "prob")[,"yes"]
evalResults$rf <- predict(caret.rf, testMatrix, type = "prob")[,"yes"]

liftData <- lift(Class ~ rf +logreg, data = evalResults, class="yes")

trellis.par.set(caretTheme())

liftplot = plot(liftData, values = 80, auto.key = list(lines = TRUE,
                                                       points = FALSE,
                                                       columns = 2))

update(liftplot, par.settings = list(fontsize = list(text = 24)))

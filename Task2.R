library(caret)
library(dplyr)
library(doParallel)
# install.packages("doParallel")
library(e1071)
# install.packages("e1071")
require(dplyr)
require(pracma)
#install.packages("pracma")

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

training <- creditDefaultData[1:25000,]
testing <- creditDefaultData[25001:30000,]

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

# Save the model for later usage
save(caret.kNN,file = "caretkNN.rda")
load("caretkNN.rda")

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

save(caret.logReg,file = "caretLogReg.rda")
load("caretLogReg.rda")

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

save(caret.ctree,file = "caretCtree.rda")
load("caretCtree.rda")

# Train a random forest
cl <- makeCluster(detectCores())
registerDoParallel(cl)
caret.rf <- train(trainMatrix,
                  training$default.payment.next.month,
                  preProcess=c("knnImpute"),
                  method = "rf",
                  metric = "Accuracy",
                  tuneLength = 10,
                  trControl = tc)
stopCluster(cl)

save(caret.rf,file = "caretRf.rda")
load("caretRf.rda")

# Provide a table akin to table 1
models = list(caret.logReg,caret.ctree,caret.kNN,caret.rf)

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

# Free up the memory
mod = NULL
models = NULL

# Area Ratio
# AUC for the Lift
evalResults <- data.frame(Class = testing$default.payment.next.month)
evalResults$logreg <-predict(caret.logReg, testMatrix, type = "prob")[,"1"]
evalResults$ctree <- predict(caret.ctree, testMatrix, type = "prob")[,"1"]
evalResults$rf <- predict(caret.rf, testMatrix, type = "prob")[,"1"]
evalResults$knn <- predict(caret.kNN, testMatrix, type = "prob")[,"1"]

evalResults %>%
  arrange(-logreg) %>%
  mutate(count=1:n()) %>%
  mutate(cumulativeHits = cumsum(.$Class=="1")) %>%
  mutate(cumulativePercentage = cumulativeHits/max(cumulativeHits),
         count = count / n())-> LiftLog
AUCLog = trapz(LiftLog$count,LiftLog$cumulativePercentage)

evalResults %>%
  arrange(-ctree) %>%
  mutate(count=1:n()) %>%
  mutate(cumulativeHits = cumsum(.$Class=="1")) %>%
  mutate(cumulativePercentage = cumulativeHits/max(cumulativeHits),
         count = count / n())-> Liftctree
AUCctree = trapz(Liftctree$count,Liftctree$cumulativePercentage)

evalResults %>%
  arrange(-knn) %>%
  mutate(count=1:n()) %>%
  mutate(cumulativeHits = cumsum(.$Class=="1")) %>%
  mutate(cumulativePercentage = cumulativeHits/max(cumulativeHits),
         count = count / n())-> Liftknn
AUCknn = trapz(Liftknn$count,Liftknn$cumulativePercentage)

evalResults %>%
  arrange(-rf) %>%
  mutate(count=1:n()) %>%
  mutate(cumulativeHits = cumsum(.$Class=="1")) %>%
  mutate(cumulativePercentage = cumulativeHits/max(cumulativeHits),
         count = count / n())-> Liftrf
AUCrf = trapz(Liftrf$count,Liftrf$cumulativePercentage)

# Area below Baseline = 0.5 
# Area below Best possible Curve
AUCbest <- ((sum(Liftctree$Class=="1")/5000)^2)/0.5+((5000-sum(Liftctree$Class=="1"))/5000)

AreRaLog <- (AUCLog-0.5)/(AUCbest-0.5)
AreRaCtree <- (AUCctree-0.5)/(AUCbest-0.5)
AreRaKnn <- (AUCknn-0.5)/(AUCbest-0.5)
AreRaRf <- (AUCrf-0.5)/(AUCbest-0.5)

AreaRatio <- c(AreRaLog,AreRaCtree,AreRaKnn,AreRaRf)

AllMetrics = data.frame(row.names = ModName,ErrRaTr,ErrRaVal,AreaRatio)

# b.

# Develop the graphs and apply to your classifiers
# Sorting smoothing for LogReg
evalResults %>%
  select(Class,logreg)%>%
  arrange(logreg)%>%
  mutate(block=rep(1:(n()/50),each=50)) %>%
  mutate(Class = ifelse(Class=="1",1,0))%>%
  group_by(block) %>%
  mutate(avg = sum(Class)/50) -> smoothLogReg

plot(smoothLogReg$avg,smoothLogReg$logreg)
abline(lm(logreg~avg,data=smoothLogReg),add=T)

evalResults %>%
  select(Class,ctree)%>%
  arrange(ctree)%>%
  mutate(block=rep(1:(n()/50),each=50)) %>%
  mutate(Class = ifelse(Class=="1",1,0))%>%
  group_by(block) %>%
  mutate(avg = sum(Class)/50) -> smoothCtree

plot(smoothCtree$avg,smoothCtree$ctree)
abline(lm(ctree~avg,data=smoothCtree),add=T)

evalResults %>%
  select(Class,knn )%>%
  arrange(knn)%>%
  mutate(block=rep(1:(n()/50),each=50)) %>%
  mutate(Class = ifelse(Class=="1",1,0))%>%
  group_by(block) %>%
  mutate(avg = sum(Class)/50) -> smoothKnn

plot(smoothKnn$avg,smoothKnn$logreg)
abline(lm(logreg~avg,data=smoothRf),add=T)

evalResults %>%
  select(Class,rf)%>%
  arrange(rf)%>%
  mutate(block=rep(1:(n()/50),each=50)) %>%
  mutate(Class = ifelse(Class=="1",1,0))%>%
  group_by(block) %>%
  mutate(avg = sum(Class)/50) -> smoothRf

plot(smoothRf$avg,smoothRf$logreg)
abline(lm(logreg~avg,data=smoothRf),add=T)

# How would you assess the performance of your random forest 
# vis a vis the methods in the paper

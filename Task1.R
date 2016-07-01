require(magrittr)
require(dplyr)
require(ggplot2)
require(class)
require(doParallel)
require(caret)
trainDataAll <- read.csv("basketball_train.csv", sep = ";")

#a
trainDataAll%>%
  select(playoffs, shot_made_flag, shot_type)-> ShotData

ShotData%>%
  filter(playoffs == 0 )%>%
  filter(shot_type == "2PT Field Goal" )%>%
  count(.)%>%
  as.numeric(.[1,1]) -> Season2Point

ShotData%>%
  filter(playoffs == 0 )%>%
  filter(shot_type == "2PT Field Goal" )%>%
  filter(shot_made_flag == 0 )%>%
  count(.)%>%
  as.numeric(.[1,1]) -> Season2PointNM
        
evalSeason2PointNM <- Season2PointNM/Season2Point 
# 52,46% Season 2-Points Not Made

ShotData%>%
  filter(playoffs == 1 )%>%
  filter(shot_type == "2PT Field Goal" )%>%
  count(.)%>%
  as.numeric(.[1,1]) -> PlayOff2Point

ShotData%>%
  filter(playoffs == 1 )%>%
  filter(shot_type == "2PT Field Goal" )%>%
  filter(shot_made_flag == 0 )%>%
  count(.)%>%
  as.numeric(.[1,1]) -> PlayOff2PointNM

evalPlayOff2PointNM <- PlayOff2PointNM/PlayOff2Point 
# 47,54% Playoffs 2-Points Not Made
  
trainDataAll %>%
  select(playoffs, shot_made_flag, shot_type)%>%
  filter(playoffs == 0 ) %>%                  
  filter(shot_type == "3PT Field Goal" ) -> Season3Point

Season3Point %>%
  filter(.,shot_made_flag == 0 ) -> Season3PointNM               
                                              
(count(Season3PointNM)/count(Season3Point)) ->  #66,73% Season 3-Points Not Made

trainData %>%
    select(playoffs, shot_made_flag, shot_type)%>%
    filter(playoffs == 1 )%>%                  
    filter(shot_type == "3PT Field Goal" ) -> Season3Point
  
  Season3Point %>%
    filter(shot_made_flag == 0 ) -> Season3PointNM               
  
  (count(Season3PointNM)/count(Season3Point)) -> #69,57 Playoffs 3-Points Not Made

#____________Spatial_visualization_________________________________
# Create bins by making intervals along the axis
# like laying a chess board on the field 
# Intervals on the y axis min. -44, max. 791 and 425 unique values
# Range on the y axis is 835.  Div by 5 makes 167 bins   
# Intervals on the x axis min. -250, max. 248 and 485 unique values
# Range on the x axis is 498, round to 500. Div by 5 makes 100 bins 

# An Outlier has been removed, for better visualization 

freq <- function(x)
{
  total_shots <- length(x)
  shots_made <- sum(x>0)
  freq_made <- shots_made/total_shots
  return(freq_made)
}
  
trainDataAll%>%
  select(loc_x,loc_y,shot_made_flag)%>%
  mutate(bin_y = round(loc_y/6),bin_x = round(loc_x/6))%>%
  group_by(bin_y,bin_x)%>%
  summarise(shot_made_freq = freq(shot_made_flag), freq_shot = ifelse(length(shot_made_flag)<1000,length(shot_made_flag),15))%>%
  ggplot(.,aes(x=bin_x, y=bin_y))+
  geom_point(aes(color = shot_made_freq,size = freq_shot))+
  scale_color_gradient2(low = "blue",high = "red",mid = "yellow", midpoint = 0.5)+
  ylim(-9,75)+
  scale_size_area()-> Graph
Graph

#____________Breaking_seconds_into_intervals_______

# Alternatively create a variable with total seconds remaining 

trainDataAll%>%
  select(period,minutes_remaining,seconds_remaining,shot_made_flag)%>%
  mutate(bin_sec15 = round((seconds_remaining+8)/15),bin_sec30 = round((seconds_remaining+16)/30))%>%
  group_by(period,minutes_remaining)%>%
  summarise(shot_count = length(shot_made_flag))%>%
  ggplot(.,aes(minutes_remaining,shot_count))+
  geom_point()+
  geom_smooth()+
  facet_grid(.~period)
  

# b kNN Classification 
# base the classification on shot location and time remaining
# use varying values of k
trainDataAll%>%
  select(loc_y,loc_x,period,minutes_remaining,seconds_remaining,shot_made_flag)%>%
  mutate(bin_sec15 = round((seconds_remaining+8)/15),bin_sec30 = round((seconds_remaining+16)/30)) -> trainData

trainData$loc_y = scale(trainData$loc_y)
trainData$loc_x = scale(trainData$loc_x)
trainData$minutes_remaining = scale(trainData$minutes_remaining)
trainData$seconds_remaining = scale(trainData$seconds_remaining)
trainData$shot_made_flag = as.factor(trainData$shot_made_flag)

train = trainData[1:16000,]
test = trainData[16001:nrow(trainDataAll),]

# Train the kNN Model solely with the location as predictor
train_1 = select(train,-period,-minutes_remaining,-seconds_remaining,-shot_made_flag)
test_1  = select(test,-period,-minutes_remaining,-seconds_remaining,-shot_made_flag)

for (k in 1:40) 
{
  summary(knn(train_1,test_1,train_1$shot_made_flag, k))
}
# The centers and the cirkels of the kNN would resemble the playing field

# Train the model with all variables

data = NULL
for (k in 1:45){
  data = c(data,sum(knn(select(train,-shot_made_flag),select(test,-shot_made_flag),train$shot_made_flag, k)!=test$shot_made_flag))
}

ts.plot(data,xlab="k",ylab="# misclassifications")

# Experimenting with different scaling of the remaining time 
# Why is this crucial for the kNN classifier? 

train_3 = select(train,-seconds_remaining,-bin_sec30,-shot_made_flag)
test_3 = select(test,-seconds_remaining,-bin_sec30,-shot_made_flag)

data = NULL
for (k in 1:50){
  data = c(data,sum(knn(train_3,test_3,train$shot_made_flag, k)!=test$shot_made_flag))
}

ts.plot(data,xlab="k",ylab="# misclassifications")

# Alternatively with bin_sec15

# c. Logistic regression

# Basic logistic regression as a benchmark
trainData = trainDataAll

trainData$shot_made_flag = as.numeric(trainData$shot_made_flag)

logReg = glm(shot_made_flag~loc_x,data = trainData,family = "binomial")
plot(trainData$loc_x,trainData$shot_made_flag, col = "blue", pch="|")
curve(predict(logReg, data.frame(loc_x=x), type="response"), add=TRUE) 

# Convert factors to factors
trainData$shot_made_flag = as.factor(trainData$shot_made_flag)
trainData$loc_y = as.factor(trainData$loc_y)
trainData$loc_x = as.factor(trainData$loc_x)
trainData$minutes_remaining = as.factor(trainData$minutes_remaining)
trainData$period = as.factor(trainData$period)
trainData$playoffs = as.factor(trainData$playoffs)
trainData$seconds_remaining = as.factor(trainData$seconds_remaining)

# Splitting the data set 
training = trainData[1:16000,]
testing = trainData[16001:nrow(trainData),]

logReg.basic = glm(shot_made_flag~loc_x,family="binomial",data=training)
summary(logReg.basic)
plot(trainData$loc_x,trainData$shot_made_flag, col = "blue", pch="|")
curve(predict(logReg.basic, data.frame(loc_x=x), type="response"), add=TRUE) 


logReg.basic <- train(shot_made_flag~minutes_remaining+period+playoffs+
                        season+seconds_remaining+shot_type,
                      data = training,
                      method = "glm",
                      family = "binomial")

#Warnings arise from full rank matrix
prediction = predict(logReg.basic,testMatrix)
confusionMatrix(prediction,testing$shot_made_flag)

trainData = trainDataAll
# Logistic Regression from polar coordinates against the shot made flag
# Coordinates converted to polar coordinates
# Function to convert the cartesian coordinates in polar
rad2deg <- function(rad) {(rad * 180) / (pi)}

cartToPolar <- function(x,y)
{
  r = round(sqrt(x^2+y^2))
  theta = round(rad2deg(atan(y/x)))
  theta = ifelse(is.nan(theta),0,theta)
  polarCoord = paste(as.character(theta),".",as.character(r))
  polarCoord = as.factor(gsub(" ","",polarCoord))
  return(c(r,theta))
}

polarDist <- function(x,y)
{
  r = round(sqrt(x^2+y^2))
  return(r)
}

polarAngl <- function(x,y)
{
  theta = round(rad2deg(atan2(y,x)))
  theta = ifelse(is.nan(theta),0,theta)
  return(theta)
}
trainData%>%
  mutate(polarDista = polarDist(loc_x,loc_y), polarAngle = polarAngl(loc_x,loc_y))-> trainData

training = trainData[1:16000,]
testing = trainData[16001:nrow(trainData),]

cl <- makeCluster(detectCores())
registerDoParallel(cl)
logReg.polarDista <- train(shot_made_flag~polarDista,
                      data = training,
                      method = "glm",
                      family = "binomial")
stopCluster(cl)

plot(trainData$polarDista,trainData$shot_made_flag, col = "blue", pch="|")
curve(predict(logReg.polarDista$finalModel,data.frame(polarDista=x), type="response"), add=TRUE) 

cl <- makeCluster(detectCores())
registerDoParallel(cl)
logReg.polarangl <- train(shot_made_flag~polarAngle,
                      data = training,
                      method = "glm",
                      family = "binomial")
stopCluster(cl)
logReg.polarangl$finalModel

plot(trainData$polarAngle,trainData$shot_made_flag, col = "blue", pch="|")
curve(predict(logReg.polarangl$finalModel,data.frame(polarAngle=x), type="response"), add=TRUE) 

cl <- makeCluster(detectCores())
registerDoParallel(cl)
logReg.polar <- train(shot_made_flag~polarAngle+polarDista,
                  data = training,
                  method = "glm",
                  family = "binomial")
stopCluster(cl)

logReg.polar$finalModel

# Attacking from the left or from the right of the basket
# What angles are on the left and what angles are on the right
angles = NULL
for (y in -2:2) 
{
  for (x in -2:2) 
    {
    angle = round(rad2deg(atan2(y,x)))
    angle = cbind(y,x,angle)
    angles= rbind(angles,angle)
    }
}
# The "left attack-angles" are from 180 to 90 in the positive range and -90 to -180
# The "right attack-angles" are from 0 to 90 and from 0 to -90 

trainData%>%
  mutate(attack= as.factor(ifelse((.$polarAngle >90 &.$polarAngle < 180)|(.$polarAngle <(-90)),"l","r"))) -> trainData
training = trainData[1:16000,]
testing = trainData[16001:nrow(trainData),]

plot(trainData$attack,trainData$shot_made_flag, col = "blue", pch="|")

cl <- makeCluster(detectCores())
registerDoParallel(cl)
logReg.polarAttack <- train(shot_made_flag~attack+polarDista,
                      data = training,
                      method = "glm",
                      family = "binomial")
stopCluster(cl)

logReg.polarAttack$finalModel

# Crunch time (remaining Time < X)last few minutes in the fourth quarter
trainData%>%
  mutate(crunchTime = ifelse(period == 4 & minutes_remaining <= 3, 1,0))-> trainData

training = trainData[1:16000,]
testing = trainData[16001:nrow(trainData),]

cl <- makeCluster(detectCores())
registerDoParallel(cl)
logReg.polarAttackCrunch <- train(shot_made_flag~attack+polarDista+crunchTime,
                            data = training,
                            method = "glm",
                            family = "binomial")
stopCluster(cl)

logReg.polarAttackCrunch$finalModel

# Aging Curve
trainData%>%
  mutate(year = strsplit(as.character(season),"-"))%>%
  mutate(year = as.numeric(year[[1]][1])) -> trainData

# Favourite Spots to score 
trainData%>%
  mutate(dist_bin = round(polarDista/6),angle_bin = round(polarAngle/5))-> test

# pick the most promising model

# d Train a random forest model 

# e

# The rationale behind this metric is to create a count of the misclassified observations. 
# Where a misclassified observation gets the value 1 and and a right classification gets a 0.
# Hence, high values indicate a low classification accuracy of the implemented model.

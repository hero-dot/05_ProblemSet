library(magrittr)
library(dplyr)
library(ggplot2)

trainDataAll <- read.csv("basketball_train.csv", sep = ";")

trainData <- head(trainDataAll,16000)

#a
#

trainData %>%
  select(playoffs, shot_made_flag, shot_type)%>%
  filter(.,playoffs == 0 ) %>%                 
  filter(.,shot_type == "2PT Field Goal" ) -> Season2Point   

  Season2Point %>%
   filter(.,shot_made_flag == 0 ) -> Season2PointNM
        
(count(Season2PointNM)/count(Season2Point)) #52,46% Season 2-Points Not Made

  trainData %>%
    select(playoffs, shot_made_flag, shot_type)%>%
    filter(.,playoffs == 1 ) %>%                 
    filter(.,shot_type == "2PT Field Goal" ) -> Playoff2Point   
  
  Season2Point %>%
    filter(.,shot_made_flag == 0 ) -> Playoff2PointNM
  
  (count(Playoff2PointNM)/count(Playoff2Point)) #47,54% Playoffs 2-Points Not Made
  
trainData %>%
  select(playoffs, shot_made_flag, shot_type)%>%
  filter(.,playoffs == 0 ) %>%                  
  filter(.,shot_type == "3PT Field Goal" ) -> Season3Point

  Season3Point %>%
    filter(.,shot_made_flag == 0 ) -> Season3PointNM               
                                              
  (count(Season3PointNM)/count(Season3Point)) ->  #66,73% Season 3-Points Not Made

  trainData %>%
    select(playoffs, shot_made_flag, shot_type)%>%
    filter(.,playoffs == 1 )%>%                  
    filter(.,shot_type == "3PT Field Goal" ) -> Season3Point
  
  Season3Point %>%
    filter(.,shot_made_flag == 0 ) -> Season3PointNM               
  
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

#____________Breaking_into_minutes

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
  select(loc_y,loc_x,period,minutes_remaining,seconds_remaining,shot_made_flag) -> trainData

train = trainData[1:16000,]
test = trainData[16001:length(trainDataAll),]

data = NULL
for (k in 1:40){
  data = c(data,sum(knn(select(train,-shot_made_flag),select(test,-shot_made_flag),train$shot_made_flag, k)!=test$shot_made_flag))
}

ts.plot(data,xlab="k",ylab="# misclassifications")

table(trainDataAll$shot_made_flag)[[1]]
table(trainDataAll$shot_made_flag)[[2]]

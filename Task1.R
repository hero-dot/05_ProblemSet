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
    filter(.,shot_type == "2PT Field Goal" ) -> Season2Point   
  
  Season2Point %>%
    filter(.,shot_made_flag == 0 ) -> Season2PointNM
  
  (count(Season2PointNM)/count(Season2Point)) #47,54% Playoffs 2-Points Not Made
  
trainData %>%
  select(playoffs, shot_made_flag, shot_type)%>%
  filter(.,playoffs == 0 ) %>%                  
  filter(.,shot_type == "3PT Field Goal" ) -> Season3Point

  Season3Point %>%
    filter(.,shot_made_flag == 0 ) -> Season3PointNM               
                                              
  (count(Season3PointNM)/count(Season3Point)) ->  #66,73% Season 3-Points Not Made

  trainData %>%
    select(playoffs, shot_made_flag, shot_type)%>%
    filter(.,playoffs == 1 ) %>%                  
    filter(.,shot_type == "3PT Field Goal" ) -> Season3Point
  
  Season3Point %>%
    filter(.,shot_made_flag == 0 ) -> Season3PointNM               
  
  (count(Season3PointNM)/count(Season3Point)) -> #69,57 Playoffs 3-Points Not Made
    

    trainData %>%
    select(shot_made_flag,loc_x, loc_y)%>%
    ggplot(.,aes(x=loc_x, y=loc_y))+
    geom_point(aes(color = shot_made_flag))
 
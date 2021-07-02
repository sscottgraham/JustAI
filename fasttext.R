# fast R vectors ----------------------------------------------------------

#load libraries 
library(fastrtext)
library(dplyr)
library(stringr)
library(tidyr)
library(readr)
library(caret)
library(irr)

#Pre-Flight 
datalist = list()
data <- read_csv("warrant-data.csv")
set.seed(1234)

for (i in 1:12){
  # Set feature of interest 
  # "BS"  "CE"  "CTE" "CTH" "IDW" "PE"  "POC" "RO"  "RHC" "S"   "SC"  "TDW"
  target <- unique(data$code)[i]
  
  # ID feature of interest 
  data$feat <- ifelse(data$code==target,target,"Other")
  
  # Get n for non-feat sample
  samp <- sum(data$feat==target)
  
  #Create feature data set
  ifelse(samp > 2500,feat_data <- data %>% filter(feat==feat) %>% sample_n(2500),feat_data <- data %>% filter(feat==target))
  
  #Create equal sample comparision dataset 
  comp_samp <- data %>%
    filter(feat=="Other")%>%
    sample_n(nrow(feat_data)) #n for non-feats sample
  
  #Combine data into training set
  train_set <- rbind(comp_samp,feat_data)
  
  train_data <- sample_n(train_set,.8*nrow(train_set))
  
  test_data <- anti_join(train_set, train_data)
  
  model_file <- build_supervised(documents = train_data$sentence,
                                 targets =train_data$feat,
                                 model_path = '//word-models//my_model',
                                 dim = 20, lr = 1, epoch = 20, wordNgrams = 2)
  model <- load_model(model_file)
  predictions <- predict(model, test_data$sentence, k=1)
  print(predictions)
  
  cm <- as.data.frame(test_data$feat)
  cm$pred <- print(predictions)
  
  cm$hum <- ifelse(cm$`test_data$feat`==target,1,0)
  cm$comp <- ifelse(str_detect(cm$pred,target),1,0)
  
  cm <- cm %>% 
    select(hum, comp)
  
  irr <- kappa2(cm)
  
  datalist[[i]] <- cbind.data.frame(unique(data$code)[i],irr$value,nrow(feat_data))
}

kappas <- do.call(rbind,datalist)
kappas

kapp_values <- data.frame()

kapp_values <- as.data.frame(unlist(unique(data$code)))
kapp_values$kapp <- datalist

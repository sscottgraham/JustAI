

# TF-IDF w/ Variable Code -------------------------------------------------
# set seed and train control = bootstrap 

# load libraries 
library(tidytext)
library(dplyr)
library(readr)
library(stringr)
library(ggplot2)
library(tidyverse)
library(tm)
library(caret)

# Run all codes -----------------------------------------------------------
#data list
datalist <- list()
set.seed(1234)
data <- read_csv("warrant-data.csv")

for (i in 1:12){
  # Pre-processing ----------------------------------------------------------
  #Set feature  | "BS"  "CE"  "CTE" "CTH" "IDW" "PE"  "POC" "RO"  "RHC" "S"   "SC"  "TDW"
  data$feat <- ifelse(data$code==unique(data$code)[i],1,0)
  
  # Get n for non-feat sample
  samp <- sum(data$feat==1)
  
  #Create feature data set
  ifelse(samp > 2500,
         feat_data <- data %>% filter(feat==1) %>% sample_n(2500), 
         feat_data <- data %>% filter(feat==1))
  
  #Create equal sample comparision dataset 
  comp_samp <- data %>%
    filter(feat==0)%>%
    sample_n(nrow(feat_data)) #n for non-feats sample
  
  #Combine data into training st
  train_set <- rbind(comp_samp,feat_data)
  
  #Check for feature non-feature equivilence 
  #train_set %>%
  #  ggplot(aes(feat)) +
  #  geom_bar()
  
  train_set$id <- seq.int(nrow(train_set))
  
  #Prepare DTM & Meta data 
  data_counts <- map_df(1:2,
                        ~ unnest_tokens(train_set, word, sentence, 
                                        token = "ngrams", n = .x)) %>%
    anti_join(stop_words, by = "word") %>%
    count(id, word, sort = TRUE)
  
  words_10 <- data_counts %>%
    group_by(word) %>%
    summarise(n = n()) %>% 
    filter(n >= 10) %>%
    select(word) %>%
    na.omit()
  
  data_dtm <- data_counts %>%
    right_join(words_10, by = "word") %>%
    bind_tf_idf(word, id, n) %>%
    cast_dtm(id, word, tf_idf)
  
  meta <- tibble(id = as.numeric(dimnames(data_dtm)[[1]])) %>%
    left_join(train_set[!duplicated(train_set$id), ], by = "id")
  
  #meta$war <- as.factor(as.character(meta$war))
  
  # Prepare Training Sets 
  #set.seed(1234)
  trainIndex <- createDataPartition(meta$feat, p = 0.8, list = FALSE, times = 1)
  
  data_df_train <- data_dtm[trainIndex, ] %>% as.matrix() %>% as.data.frame()
  data_df_test <- data_dtm[-trainIndex, ] %>% as.matrix() %>% as.data.frame()
  
  #data_df_train <- na.omit(data_df_train)
  response_train <- meta$feat[trainIndex]
  
  # View missing data
  #train_set %>%
  #  anti_join(meta, by = "id") %>%
  #  head(25) %>%
  #  pull(sentence)
  
  trctrl <- trainControl(method = "boot")
  
  svm_mod <- train(x = data_df_train,
                   y = as.factor(response_train),
                   method = "svmLinearWeights2",
                   trControl = trctrl,
                   tuneGrid = data.frame(cost = 1, 
                                         Loss = 0, 
                                         weight = 1))
  
  svm_pred <- predict(svm_mod,
                      newdata = data_df_test)
  svm_cm <- confusionMatrix(svm_pred, as.factor(meta[-trainIndex, ]$feat))
  
  datalist[[i]] <- cbind.data.frame(unique(data$code)[i],svm_cm$overall[2],nrow(feat_data))
}

kappas <- do.call(rbind,datalist)
kappas
  



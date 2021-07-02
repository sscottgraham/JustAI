
# spaCy aveloc variables ----------------------------------------------


#load libraries 
library(spacyr)
library(dplyr)
library(stringr)
library(tidyr)
library(readr)
library(fuzzyjoin)
library(reticulate)
library(caret)


# Initialize spaCy & virtualenv -------------------------------------------
spacy_initialize(model = "en_core_web_lg")


# De-initialize and kill virtualenv ---------------------------------------
spacy_finalize()


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
  
  train_set$id <- seq.int(nrow(train_set))
  
  #parse with spaCy and cosolidate entities
  data_parsed <- spacy_parse(train_set$sentence, lemma = FALSE, dependency = TRUE) %>% 
    filter(pos != "PUNCT") %>% 
    group_by(doc_id, pos) %>% 
    summarise(ave_loc = mean(token_id),
              ave_head_loc = mean(head_token_id)) %>% 
    pivot_longer(cols = ave_loc:ave_head_loc, names_to = "var", values_to = "loc") %>% 
    pivot_wider(names_from = c(pos,var),values_from = loc) %>% 
    replace(is.na(.), 0) %>%
    mutate(doc_order = as.numeric(str_replace(doc_id,"text",""))) %>% 
    arrange(doc_order) %>% 
    cbind.data.frame(train_set)
  
  data_features <-
    data_parsed %>% 
    select(id,ADJ_ave_loc:SYM_ave_head_loc)
  
  # Training and Classification  --------------------------------------------
  
  # Prepare Training Sets 
  trainIndex <- createDataPartition(data_parsed$feat, p = 0.8, list = FALSE, times = 1)
  
  data_df_train <- data_features[trainIndex, ]
  data_df_test <- data_features[-trainIndex, ]
  
  #data_df_train <- na.omit(data_df_train)
  response_train <- data_parsed$feat[trainIndex]
  
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
  svm_cm <- confusionMatrix(svm_pred, as.factor(data_parsed[-trainIndex, ]$feat))

  datalist[[i]] <- cbind.data.frame(unique(data$code)[i],svm_cm$overall[2],nrow(feat_data))
}


kappas <- do.call(rbind,datalist)
kappas

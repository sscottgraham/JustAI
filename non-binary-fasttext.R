# FastText non-binary -----------------------------------------------------


# fast R vectors ----------------------------------------------------------

#load libraries 
library(fastrtext)
library(dplyr)
library(stringr)
library(tidyr)
library(readr)
library(caret)
library(irr)


#load data
data <- read_csv("warrant-data.csv")


# Equalize by code --------------------------------------------------------
# View n by code
data %>% 
  count(code) %>% 
  arrange(n) 

# Get sample size 
samp <- data %>% 
          count(code) %>% 
          arrange(n) %>% 
          filter(n > 1000) %>% 
          slice(1) %>% 
          select(n) %>% 
          as.numeric()


# equalize grops
train_set <- data %>% 
  filter(code != "TDW" & code !="POC") %>% 
  group_by(code) %>% 
  sample_n(samp) %>% 
  ungroup()


# View n by code
train_set %>% 
  count(code) %>% 
  arrange(n) 

# Partition training and testing data 
train_data <- sample_n(train_set,.8*nrow(train_set))
  
test_data <- anti_join(train_set, train_data)

#Build mdoel   
model_file <- build_supervised(documents = train_data$sentence,
                                 targets =train_data$code,
                                 model_path = '//word-models//my_model',
                                 dim = 20, lr = 1, epoch = 20, wordNgrams = 2)

# Load model 
model <- load_model(model_file)

# Predict codes on test data
predictions <- predict(model, test_data$sentence, k=1)
print(names(predictions))

#make confusion matrix   
cm <- as.data.frame(test_data$code)
cm$pred <- as.character(lapply(predictions, names))

# get pooled kapaa
kappa2(cm)



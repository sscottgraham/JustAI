# BERT FEATURE ENGINEERING | Uses MPI Cluster created via Slurm Shell


# Load Libraries ----------------------------------------------------------

library(snow)
library(doParallel)
library(dplyr)
library(readr)
library(RBERT)


# Load Data ---------------------------------------------------------------
data <- read_csv("warrant-data.csv")


# Load pretrained BERT model 
BERT_PRETRAINED_DIR <- RBERT::download_BERT_checkpoint(
  model = "bert_base_uncased"
)

#Get cluster
cl <- getMPIcluster()
registerDoParallel(cl)

# Extract Features --------------------------------------------------------

# Create sentence representations -----------------------------------------
  r <- foreach (i=1:55539, .combine=rbind.data.frame, .packages = c("dplyr","RBERT")) %dopar% { 
    BERT_feats <- extract_features(
      examples = data$sentence[i],
      ckpt_dir = BERT_PRETRAINED_DIR,
      layer_indexes = 1:12)
    
    t(BERT_feats$output %>%
        dplyr::filter(
          sequence_index == 1, 
          token == "[CLS]", 
          layer_index == 12) %>% 
        dplyr::select(dplyr::starts_with("V")) %>% 
        unlist())
  }
  r$id <- seq.int(1,55539)
  saveRDS(r, file = "bert-features.rds")


# Stop cluster
stopCluster(cl)






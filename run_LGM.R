library(lcmm)
library(reshape2)
library(parallel)  # Load parallel package

basepath = "/home/ole/projects/PAF_reanalysis"

all_pain_df_train = read.csv(paste0(basepath,'/data/X_train.csv'))

# first reshape the data into data long
all_pain_df_long_train = melt(data = all_pain_df_train[, 1:14],
                              id.vars = c("ID"),
                              variable.name = "t",
                              value.name = "y")

# form the latent growth models
lcga1 <- hlme(y ~ t, subject = "ID", ng = 1, data = all_pain_df_long_train, verbose = FALSE)

# Use 8 cores for gridsearch
num_cores <- min(16, detectCores())  # Ensure we don't exceed available cores
lcga2 <- gridsearch(rep = 8, maxiter = 100, minit = lcga1, 
                    hlme(y ~ t, subject = "ID", ng = 2, data = all_pain_df_long_train, 
                         mixture = ~ t, verbose = FALSE), cl = num_cores)

# form classes of high and low pain sensitive 
class_1_ID = lcga2$pprob[order(lcga2$pprob$prob1)[1:40],c(1)]
class_2_ID = lcga2$pprob[order(lcga2$pprob$prob1)[61:100],c(1)]
classA = all_pain_df_train[all_pain_df_train$ID %in% class_1_ID,]
classB = all_pain_df_train[all_pain_df_train$ID %in% class_2_ID,]
if (mean(apply(classA[, c(2:11)], 2, mean)) < mean(apply(classB[, c(2:11)], 2, mean))) {
  # if subjects in classA has lower average pain, then prob1 corresponds to the high pain severity class
  ID_low_LGM = class_1_ID
  ID_high_LGM = class_2_ID
  # locate the probability threshold for low and high pain severity, in terms of the high probability
  low_threshold = lcga2$pprob[order(lcga2$pprob$prob1)[41],c(3)]
  high_threshold = lcga2$pprob[order(lcga2$pprob$prob1)[60],c(3)]
}else {
  # if subjects in classA has higher average pain, then prob1 corresponds to the low pain severity class
  ID_low_LGM = class_2_ID
  ID_high_LGM = class_1_ID
  # locate the probability threshold for low and high pain severity, in terms of the high probability
  low_threshold = lcga2$pprob[order(lcga2$pprob$prob2)[41],c(4)]
  high_threshold = lcga2$pprob[order(lcga2$pprob$prob2)[60],c(4)]
}
# save
df_ID_LGM_train = data.frame(low = ID_low_LGM, high = ID_high_LGM)
# Create the ID and class column
low_class_df <- data.frame(ID = df_ID_LGM_train$low, class = 0)
high_class_df <- data.frame(ID = df_ID_LGM_train$high, class = 1)

# Combine both dataframes into one
df_combined <- rbind(low_class_df, high_class_df)
write.csv(df_combined, file = paste0(basepath, '/data/Y_train.csv'), row.names = FALSE)

# now apply to LGM model to test data 
all_pain_df_test = read.csv(paste0(basepath,'/data/X_test.csv'))

all_pain_df_long_test = melt(data = all_pain_df_test[, 1:14],
                             id.vars = c("ID"),
                             variable.name = "t",
                             value.name = "y")
# prediction
predicted_values <- predictClass(lcga2, newdata = all_pain_df_long_test)
class_1_ID_test = predicted_values[order(predicted_values$prob1)[1:20],c(1)]
class_2_ID_test = predicted_values[order(predicted_values$prob1)[31:50],c(1)]
classA_test = all_pain_df_test[all_pain_df_test$ID %in% class_1_ID_test,]
classB_test = all_pain_df_test[all_pain_df_test$ID %in% class_2_ID_test,]
if (mean(apply(classA_test[, c(2:11)], 2, mean)) < mean(apply(classB_test[, c(2:11)], 2, mean))) {
  # true if classA (1) is low, classB (2) is high
  # prob1 is the probability corresponds to high pain severity
  ID_low_LGM_test = predicted_values$ID[predicted_values$prob1<low_threshold]
  ID_high_LGM_test = predicted_values$ID[predicted_values$prob1>high_threshold]
}else{
  # false if classB (2) is low, classA (1) is high
  # prob2 is the probability corresponds to high pain severity
  ID_low_LGM_test = predicted_values$ID[predicted_values$prob2<low_threshold]
  ID_high_LGM_test = predicted_values$ID[predicted_values$prob2>high_threshold]
}
# Create a vector with the class labels
class_labels <- c(rep(1, length(ID_high_LGM_test)),
                  rep(0, length(ID_low_LGM_test)))
# Create a vector with all the IDs
test_IDs <- c(ID_high_LGM_test, ID_low_LGM_test)
# Combine the ID and class vectors into a dataframe
df_ID_LGM_test <- data.frame(ID = test_IDs, class = class_labels)

write.csv(df_ID_LGM_test, file = paste0(basepath, '/data/Y_test.csv'), row.names = FALSE)
















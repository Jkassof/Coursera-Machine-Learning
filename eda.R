library(caret)
library(dplyr)
set.seed(311)
NAs <- c("", "#DIV/0!", "NA")
pml <- read.csv("pml-training.csv", na.strings = NAs)
fin <- apply(apply(pml, 2, is.na), 2, sum) # Count number of NAs in each column
good.vars.names <- names(fin[fin<10000]) # Make a vector of good column names, used to subet

pml.df <- tbl_df(pml)
pml.pruned <- select(pml.df, one_of(good.vars.names)) %>% select(-(1:7))
train.index <- createDataPartition(y = pml.pruned$classe, p = 0.7, list = FALSE)
pml.train <- pml.pruned[train.index,]
pml.test <- pml.pruned[-train.index,]
fitControl <- trainControl(
     method = "repeatedcv",
     number = 10,
     repeats = 1)

mod1 <- train(classe ~ ., data = pml.train, method = 'rf', trControl = fitControl)

pm.test.preds <- predict(mod1, pml.test)

confusionMatrix(pm.test.preds, pml.test$classe)

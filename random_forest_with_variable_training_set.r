# This uses random forest to recognize wine, testing with variable training set size

# setwd('D:/L/source/Wine') # only for my computer

library(randomForest)

data <- read.csv('wine.data', header = F)

for ( i in seq(1, 40, by=1) )
{
    train_indices <- c(1:(1+i), 60:(60+i), 131:(131+i)) # use i test cases to train

    train_set <- data[train_indices,]
    train_features <- train_set[, -1]
    train_labels <- train_set[, 1]
    test_set <- data[-train_indices,]
    test_features <- test_set[, -1]
    test_labels <- test_set[, 1]

    rf <- randomForest(train_features, as.factor(train_labels))
    test.predict <- predict(rf, test_features)
    accuracy <- sum(test.predict == test_labels)/length(test_labels)

    cat("[", i, "] training cases for each class gives accuracy = ", accuracy, "\n")
}
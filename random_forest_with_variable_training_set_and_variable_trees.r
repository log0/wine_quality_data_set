# This uses random forest to recognize wine, testing with variable training set size and 
# variable number of trees

# setwd('D:/L/source/Wine') # only for my computer

library(randomForest)

data <- read.csv('wine.data', header = F)

for ( j in seq(10, 100, by=10) )
{
    for ( i in seq(1, 40, by=1) )
    {
        train_indices <- c(1:(1+i), 60:(60+i), 131:(131+i)) # use i test cases to train

        train_set <- data[train_indices,]
        train_features <- train_set[, -1]
        train_labels <- train_set[, 1]
        test_set <- data[-train_indices,]
        test_features <- test_set[, -1]
        test_labels <- test_set[, 1]

        rf <- randomForest(train_features, as.factor(train_labels), nTrees = j)
        test.predict <- predict(rf, test_features)
        accuracy <- sum(test.predict == test_labels)/length(test_labels)

        cat("[", j, "] trees : [", i, "] training cases for each class gives accuracy = ", accuracy, "\n")
    }
    cat("\n")
}
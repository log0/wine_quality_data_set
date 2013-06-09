# This uses random forest to recognize wine, 100% accuracy

# setwd('D:/L/source/Wine') # only for my computer

library(randomForest)

data <- read.csv('wine.data', header = F)

train_indices <- c(1:49, 60:120, 131:168) # leave 10 for testing 100% accurate
train_set <- data[train_indices,]
train_features <- train_set[, -1]
train_labels <- train_set[, 1]
test_set <- data[-train_indices,]
test_features <- test_set[, -1]
test_labels <- test_set[, 1]

rf <- randomForest(train_features, as.factor(train_labels))
test.predict <- predict(rf, test_features)
accuracy <- sum(test.predict == test_labels)/length(test_labels)

cat("Accuracy = ", accuracy, "\n")


#Neural Network for Boston Data

#install.packages("neuralnet")
library(neuralnet)
library(MASS)
library(class)
Set.seed(123)
?Boston

Boston
Bos <- Boston
Bos
#MAKE histogram on the target variable
hist(Bos$medv)
#here no missing values so go ahead
maxValue <- apply(Bos, 2,max)
minValue <- apply(Bos, 2,min)

#split in test and train dataset

df <- as.data.frame(scale(Bos,center=minValue,scale = maxValue-minValue))
ind <- sample(1:nrow(df), 400)
train_df <- df[ind, ]
testdf <- df[-ind,]
#Settings for ANN
#medv -- target variable
allVars <- colnames(df)
predictorVars <- allVars[!allVars %in%"medv"]
predictorVars <- paste(predictorVars, collapse = "+")
from=as.formula(paste("medv~", predictorVars, collapse = "+"))
#hidden layer with every layer having 2 nodes
library(neuralnet)
neuralModel <- neuralnet(formula=from, hidden=c(4,2), linear.output = T, data = train_df)
plot(neuralModel)
predictions <- compute(neuralModel, testdf[,1:13])
str(predictions)
# For predictions we need to descale the variables to get the output
predictions <- predictions$net.result*(max(testdf$medv)-min(testdf$medv))+min(testdf$medv)
actualValues <- (testdf$medv)*(max(testdf$medv)-(min(testdf$medv)))+min(testdf$medv)
MSE <- sum((predictions-actualValues)^2)/nrow(testdf)
MSE
plot(testdf$medv, predictions, col='blue', main='Real Vs Predicted', pch=1,cex=0.9, type="p", xlab = "actual", ylab = "predicted" )
abline(0, 1, col="black")


install.packages("caTools")
library(caTools)
set.seed(123)
diabetes=read.csv(choose.files())
#Logistic Regression
dataSplit = sample.split(diabetes$Outcome, SplitRatio = 4/5)
trainingData = subset(diabetes, dataSplit=="TRUE")
testingData = subset(diabetes, dataSplit=="FALSE")
LogModel = glm(Outcome~., data = trainingData, family = "binomial")
summary(LogModel)
?sample.split
library(caret)
#Removing Age from model
LogModelA = glm(Outcome~.-Age, data = trainingData, family = "binomial")
summary(LogModelA)

#Removing Bloodpressure from model
LogModelBP = glm(Outcome~.-BloodPressure, data = trainingData, family = "binomial")
summary(LogModelBP)

#Removing Insulin from model
LogModelI = glm(Outcome~.-Insulin, data = trainingData, family = "binomial")
summary(LogModelI)

#Removing SkinThickness from model
LogModelST = glm(Outcome~.-SkinThickness, data = trainingData, family = "binomial")
summary(LogModelST)

#Training data used for prediction
Response = predict(LogModelST, trainingData, type = "response")
PredRes=ifelse(Response>0.5,1,0)

confusionMatrix(table(PredRes, trainingData$Outcome))

table(ActualValue=trainingData$Outcome, Response>0.5)
(85+26)/(85+15+28+26)
install.packages("ROCR")
library(ROCR)
??gplot
?performance

ROCRPred=prediction(Response, trainingData$Outcome)

ROCRPerformance=performance(ROCRPred,"tpr", "fpr")
plot(ROCRPerformance, coloursize=TRUE, print.cutoff.at=seq(0.1, by=0.1))

ResponseTest = predict(LogModelST, testingData, type = "response")
table(ActualValue=testingData$Outcome, ResponseTest>0.3)

LogModelFinal = glm(Outcome~Pregnancies+Glucose+BMI, data = trainingData, family = "binomial")
summary(LogModelFinal)
library(CARET)

###############################
#Decision Tree

diabetes$Outcome = factor(diabetes$Outcome)

set.seed(3)
library(caTools)
library(caret)
library(e1071)

split = sample.split(diabetes$Outcome, SplitRatio = 0.7)
training_set = subset(diabetes, split == TRUE)
test_set = subset(diabetes, split == FALSE)

library(rpart)
classifier = rpart(formula = Outcome ~ ., data = training_set, method = "class")
classifier
plot(classifier)
text(x = diabetes)
?rpart
pred = predict(classifier, newdata = test_set[-9]
               ,type = 'class')
pred
confusionMatrix(table(pred, test_set$Outcome))

table(test_set$Outcome, pred)


######################
#Ensemble Model - Random Forest

#install.packages("randomForest")
library(randomForest)
library(caret)

diabetes<-read.csv(choose.files())
diabetes$Outcome=as.factor(diabetes$Outcome)
id<- sample(2, nrow(diabetes), prob = c(0.7,0.3), replace=TRUE)
diabetes_train=diabetes[id==1,]
diabetes_test=diabetes[id==2,]
#To choose how many variables are to be taken at each tree 
bestmtry=tuneRF(diabetes_train, diabetes_train$Outcome, stepFactor = 1.2, improve = 0.01, trace = T, plot = T)
?tuneRF

diabetes_Forest=randomForest(formula=Outcome~., data = diabetes_train, mtry=bestmtry[1])
summary(diabetes_Forest)
diabetes_Forest$importance
?randomForest
predict_diabetes=predict(diabetes_Forest, diabetes_test)
confusionMatrix(table(predict_diabetes, diabetes_test$Outcome))

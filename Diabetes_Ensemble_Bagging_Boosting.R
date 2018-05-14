#install.packages("RCurl")

require(RCurl)

library(RCurl)
binData<-getBinaryURL("https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip",ssl.verifypeer=FALSE)

conObj <-file("dataset_diabetes.zip", open = "wb")
writeBin(binData, conObj)
close(conObj)
files<-unzip("dataset_diabetes.zip")

diabetes<-read.csv(files[1], stringsAsFactors=FALSE)
names(diabetes)
#Step 1 of pre-processing - Remove encounter_id and patient_nbr
diabetes<-subset(diabetes,select = -c(encounter_id, patient_nbr))

diabetes[diabetes=="?"]<-NA
diabetes<-diabetes[sapply(diabetes, function(x) length(levels(factor(x, exclude = NULL)))>1)]

diabetes$readmitted<-ifelse(diabetes$readmitted=='<30', 1, 0)

outcomeName<-"readmitted"
#To drop large factors
diabetes<-subset(diabetes, select = -c(diag_1, diag_2, diag_3))

#To convert data to binary data
for(colname in charcolumns) {
  print(paste(colname, length(unique(diabetes[, colname]))))
for(newcol in unique(diabetes[, colname])) {
for(newcol in unique(diabetes[, colname])) {

if(!is.na(newcol))
diabetes[, paste0(colname, "_", newcol)]<- ifelse(diabetes[, colname]==newcol, 1, 0)
 }
diabetes<-diabetes[, setdiff(names(diabetes), colname)]
}
}

dim(diabetes)
#To remove punctuation marks
colnames(diabetes)<-gsub(x=colnames(diabetes), pattern="[[:punct:]]", replacement = "_")
diabetes<-diabetes[sapply(diabetes, function(x) length(levels(factor(x, exclude=NULL)))>1)]
#Convert NA to 0
diabetes[is.na(diabetes)]<-0


#
set.seed(1234)
split<-sample(nrow(diabetes), floor(0.5*nrow(diabetes)))
traindf<-diabetes[split,]
testdf<-diabetes[-split,]
predictorNames<-setdiff(names(traindf), outcomeName)
fit<-lm(readmitted~., data=traindf)
preds<-predict(fit, testdf[, predictorNames], se.fit = TRUE)

#install.packages("pROC")
library(pROC)

#Area under the curve
print(auc(testdf[, outcomeName], pred$fit))
print(auc(testdf[, outcomeName], preds$fit))

#install.packages("foreach")
library(foreach)

#install.packages("doParallel")  
library(doParallel)

c1<-makeCluster(8)
registerDoParallel(c1)
length_divisor<-20

predictions<-foreach(m=1:400, .combine = cbind) %dopar% {
  sampleRows<-sample(nrow(traindf), size=floor((nrow(traindf)/length_divisor)))
  fit<-lm(readmitted~., data=traindf[sampleRows,])
  predictions<-data.frame(predict(object = fit, testdf[, predictorNames], se.fit = TRUE)[[1]])
}
stopCluster(c1)
  
auc(testdf[,outcomeName], rowMeans(predictions))
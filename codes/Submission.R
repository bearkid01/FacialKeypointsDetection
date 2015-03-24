# Creating the submission file to Kaggle 
pred_cnn <- read.csv("pred_cnn.csv",header=TRUE)
dim(pred_cnn)
pred_cnn$X <- pred_cnn$X+1

library(reshape)
library(plyr)

new_pred_cnn <- melt(pred_cnn,id="X")
new_pred_cnn <- new_pred_cnn[order(new_pred_cnn$X),]

# import idLookupTable 
idLookup <- read.csv("/Users/bearkid/Downloads/IdLookupTable.csv",header=TRUE)
idLookup$Location <- NULL
colnames(new_pred_cnn) <- c("ImageId","FeatureName","Value")
new_idLookup <- join(idLookup,new_pred_cnn)
head(new_idLookup)
new_idLookup$ImageId <- NULL
new_idLookup$FeatureName <- NULL
colnames(new_idLookup) <- c("RowId","Location")
write.table(new_idLookup,"/Users/bearkid/FacialKeypointsDetection/pred_cnn_2.csv",sep=",",row.name=FALSE)


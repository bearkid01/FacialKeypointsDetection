library(bigmemory)
library(neuralnet)

regular_time<-system.time(read.csv("training.csv",header=TRUE,stringsAsFactors=F))
bigmemory_time<-system.time(read.big.matrix("training.csv",header=TRUE,type="integer"))

df<-read.big.matrix("training.csv",sep=',',header=TRUE)


df<-df[complete.cases(df[1:4]),]
### test ###
column_1 <- df[,1]
column_2 <- df[,2]


im.train<-df$Image

image_1<-as.integer(unlist(strsplit(im.train[1], " ")))
im <- matrix(data=rev(image_1), nrow=96, ncol=96)
image(1:96, 1:96, im, col=gray((0:255)/255))

# try SVD on the picture 
library(MASS)
im.svd <- svd(im)
diag_im <- diag(im.svd$d)
image(1:96, 1:96, diag_im)

u <- im.svd$u
v <- im.svd$v
d <- diag(im.svd$d)

depth <- 24 
us <- as.matrix(u[,1:depth])
vs <- as.matrix(v[,1:depth])
ds <- as.matrix(d[1:depth,1:depth])
depth_im <- us %*% ds %*% t(vs)
image(1:96, 1:96, depth_im, col=gray((0:255)/255))

ptm <- proc.time()
foreach(i=1:50000) %dopar% sqrt(i)
proc.time()-ptm

ptm <-proc.time()
for(i in 1:50000){
  print(sqrt(i))
}
proc.time()-ptm


library(doParallel)
cl <- makeCluster(4)
registerDoParallel(cl)


x <- iris[which(iris[,5] != "setosa"), c(1,5)]
trials <- 10000

ptime <- system.time({
  r <- foreach(icount(trials), .combine=cbind) %dopar% {
    ind <- sample(100, 100, replace=TRUE)
    result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
    coefficients(result1)
    }
  })[3]
ptime


stime <- system.time({
  r <- foreach(icount(trials), .combine=cbind) %do% {
    ind <- sample(100, 100, replace=TRUE)
    result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
    coefficients(result1)
    }
  })[3]

stime

# borrowed this function from http://stackoverflow.com/questions/10865489/scaling-an-r-image
"
resizePixels = function(im, w, h) {
  pixels = as.vector(im)
  # initial width/height
  w1 = nrow(im)
  h1 = ncol(im)
  # target width/height
  w2 = w
  h2 = h
  # Create empty vector
  temp = vector('numeric', w2*h2)
  # Compute ratios
  x_ratio = w1/w2
  y_ratio = h1/h2
  # Do resizing
  for (i in 0:(h2-1)) {
    for (j in 0:(w2-1)) {
      px = floor(j*x_ratio)
      py = floor(i*y_ratio)
      temp[(i*w2)+j] = pixels[(py*w1)+px]
    }
  }
  
  m = matrix(temp, h2, w2)
  return(m)
}
"

image(1:48,1:48,resizePixels(im,48,48),col=gray((0:255)/255))
image(1:96,1:96,resizePixels(im,96,96),col=gray((0:255)/255))
left_eye_x <- df[,1][1]
left_eye_y <- df[,2][1]
right_eye_x <- df[,3][1]
right_eye_y <- df[,4][1]
updated_left_eye_x <- df[,1][1]/2
updated_left_eye_y <- df[,2][1]/2
updated_right_eye_x <- df[,3][1]/2
updated_right_eye_y <- df[,4][1]/2

# 96 - 
points(96-left_eye_x,96-left_eye_y)
points(96-right_eye_x,96-right_eye_y)

# 48 - 
points(48-updated_left_eye_x,48-updated_left_eye_y,col="red")
points(48-updated_right_eye_x,48-updated_right_eye_y,col="red")

# randomly choose how many images to test 
df<-read.csv("training.csv",header=TRUE,stringsAsFactors=F,na.rm=TRUE)
df<-na.omit(df)
num_rows <- nrow(df)
chosen_indicies <- sample(1:num_rows,2000)
chosen_indicies 
# indicies that are not selected 
left_indicies <- setdiff(1:num_rows,chosen_indicies)
# pull the images 
chosen_images <- df[chosen_indicies,]
chosen_images_test <-df[left_indicies,]
dim(chosen_images)
images <- chosen_images[,31]

image_list <- c()
for(i in 1:nrow(chosen_images)){
  im<- as.integer(unlist(strsplit(images[i]," ")))
  im_matrix <- matrix(data=rev(im),nrow=96,ncol=96)
  im_matrix_updated<-as.vector(resizePixels(im_matrix,24, 24))  
  image_list <- c(image_list,data.frame(pixels=im_matrix_updated))
}

wide_images <- do.call("rbind", image_list)
rownames(wide_images)<-NULL
col_names <- paste("X",1:ncol(wide_images),sep="")
wide_df_images <- data.frame(wide_images)
colnames(wide_df_images) <- col_names

# scale the data 
#wide_df_images <- scale(wide_df_images,center=F)

half<-sapply(chosen_images[,c(1:4,21:22)],function(x){x/4})
wide_df_images<-cbind(wide_df_images,half)

# think of the ways to represent X1,X2,X3,X4,X5,,,,

xnam <- paste0("X", 1:ncol(wide_images))
#fmla <- as.formula(paste("left_eye_center_x+left_eye_center_y+right_eye_center_x+right_eye_center_y+nose_tip_x+nose_tip_y ~ ", 
                         #paste(xnam, collapse= "+")))
fmla <- as.formula(paste("cbind(left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y,nose_tip_x,nose_tip_y)"," ~ ", 
                         paste(xnam, collapse= "+")))

mod <- nnet(fmla, data=wide_df_images,size=2,linout=T,skip=TRUE, MaxNWts=10000,maxit=5000, decay=0.2)
dim(mod$residuals)
#mod <- neuralnet(fmla,data=wide_df_images,hidden=3,learningrate=0.01)
plot.nnet(mod)
plot(mod)

# prepare the testing set and calculate the Root Mean Square Error 
im_test<- as.integer(unlist(strsplit(chosen_images_test[,31]," ")))
im_matrix_test <- matrix(data=rev(im_test),nrow=96,ncol=96)
im_matrix_updated_test<-as.vector(resizePixels(im_matrix_test,24, 24))  
test <- matrix(im_matrix_updated_test,nrow=1,ncol=24*24)
test<-data.frame(test)
colnames(test)<-col_names

# prepare the testing set and calculate the Root Mean Square Error 
test_rbind<-foreach(i = 1:nrow(chosen_images_test),.combine=rbind) %dopar% {
  im_test<- as.integer(unlist(strsplit(chosen_images_test[i,31]," ")))
  im_matrix_test <- matrix(data=rev(im_test),nrow=96,ncol=96)
  im_matrix_updated_test<-as.vector(resizePixels(im_matrix_test,24,24))  
  test <- matrix(im_matrix_updated_test,nrow=1,ncol=24*24)
  data.frame(test)
}

rownames(test_rbind)<-NULL
test_rbind<-data.frame(test_rbind)
colnames(test_rbind)<-col_names

#############################################################################
pred <- predict(mod,newdata=test_rbind)

# im_test_output<-sapply(chosen_images_test[,c(1:4,21:22)],function(x){x/2})
# computed <- as.vector(compute(mod,test)$net.result)
# result_df <- data.frame(true=im_test_output,computed=computed)
# result_df$true <- 2*result_df$true
# result_df$computed <- 2*result_df$computed
# result_df$diff <- result_df$computed-result_df$true
# sqrt(mean(result_df$diff^2))

im_test_output<-sapply(chosen_images_test[,c(1:4,21:22)],function(x){x/4})
#computed <- compute(mod,test_rbind)$net.result
#computed <- as.vector(compute(mod,test_rbind)$net.result)
colnames(pred) <- c("left_eye_center_x_output","left_eye_center_y_output","right_eye_center_x_output",
                        "right_eye_center_y_output","nose_tip_x_output","nose_tip_y_output")

# RMSE, still not sure about the equation 
combined <- data.frame(cbind(im_test_output,pred))
combined[,1:12] <- lapply(combined[,1:12],function(x){x*4})
pred <- combined[,7:12]
im_test_output <- combined[,1:6]
sqrt(mean((im_test_output-pred)^2, na.rm=T))

# visualize on the resized picture to see if it is close 
image(1:48, 1:48,resizePixels(im_matrix_test,48, 48),col=gray((0:255)/255))
points(48-result_df[1,1],48-result_df[2,1])
points(48-result_df[3,1],48-result_df[4,1])
points(48-result_df[1,2],48-result_df[2,2],col="red")
points(48-result_df[3,2],48-result_df[4,2],col="red")

# visualize on the original picture to see if it is close 
image(1:96,1:96,im_matrix_test,col=gray((0:255)/255))
keypoints <- chosen_images_test[,c(1:4,21:22)]
points(96-keypoints[1,1],96-keypoints[1,2])
points(96-keypoints[1,3],96-keypoints[1,4])
points(96-keypoints[1,5],96-keypoints[1,6])
points(96-result_df[1,2],96-result_df[2,2],col="red")
points(96-result_df[3,2],96-result_df[4,2],col="red")
points(96-result_df[5,2],96-result_df[6,2],col="red")


# visualize the locations of facial keypoints

df_images <- df$Image
image_list_duped <- c()
for(i in 1:nrow(df)){
  im<- as.integer(unlist(strsplit(df_images[i]," ")))
  image_list_duped <- c(image_list_duped,data.frame(pixels=im))
}

rBind <- do.call("rbind",image_list_duped)
rownames(rBind)<-NULL

#left eye 
library(ggplot2)
left_eye<-ggplot(df,aes(x=96-left_eye_center_x,y=96-left_eye_center_y))+geom_point(color="orange",size=2,na.rm=TRUE)
left_eye

#right eye
right_eye<-ggplot(df,aes(x=96-right_eye_center_x,y=96-right_eye_center_y))+geom_point(color="blue",size=2,na.rm=TRUE)
right_eye

#nose tip 
nose_tip <- ggplot(df,aes(x=96-nose_tip_x,y=96-nose_tip_y))+geom_point(color="green",size=2,na.rm=TRUE)
nose_tip

#left eye, right eye and nose tip 
three_keypoints<-ggplot(df)+ geom_point(aes(x=96-nose_tip_x,y=96-nose_tip_y,color="Nose Tip"),size=2,na.rm=TRUE) + geom_point(aes(x=96-right_eye_center_x,y=96-right_eye_center_y,color="Right Eye"),size=2,na.rm=TRUE) + geom_point(aes(x=96-left_eye_center_x,y=96-left_eye_center_y,color="Left Eye"),size=2,na.rm=TRUE)
three_keypoints<-three_keypoints+ggtitle("Scatterplot of the Locations of Three Facial Keypoints")
three_keypoints<-three_keypoints+xlab("X")+ylab("Y")
three_keypoints<-three_keypoints+scale_colour_discrete(name = "Type")
three_keypoints

# the pixel density around left eye, right eye and nose tip 

# kaggle

d  <- read.csv("training.csv", stringsAsFactors=F)
im <- foreach(im = d$Image, .combine=rbind) %dopar% {
  as.integer(unlist(strsplit(im, " ")))
}
d$Image <- NULL
set.seed(0)
idxs     <- sample(nrow(d), nrow(d)*0.8)
d.train  <- d[idxs, ]
d.test   <- d[-idxs, ]
im.train <- im[idxs,]
im.test  <- im[-idxs,]
rm("d", "im")


p <- matrix(data=colMeans(d.train, na.rm=T), nrow=nrow(d.test), ncol=ncol(d.train), byrow=T)

sqrt(mean((d.test-p)^2, na.rm=T))




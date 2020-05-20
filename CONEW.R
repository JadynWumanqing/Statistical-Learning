rm(list=ls())
cat("\014")
library(randomForest)
library(ggplot2)
library(gridExtra)
library(glmnet)

#Data Source: https://data.world/usafacts/coronavirus-in-the-us
#Background: The dataset contains confirmed Covid-19 cases by U.S. counties from January 22nd, 2020 to May 18th, 2020.


setwd(getwd())
co.ori  =  na.omit(read.csv("covid_confirmed_usafacts.csv",header=TRUE))
#View(co.ori)
co      =  co.ori[-1,-c(1,2,3,4)] #deletting row 1: no a data point
                          #excluding column 1 through 4: geographical information
set.seed(1)

n       =  dim(co)[1] #3194
p       =  dim(co)[2]

for (i in 1:n) {
  co[i,] = c(0,diff(as.numeric(co[i,])))
}


y        =  co[,p]
X        =  data.matrix(co[,-p]) #3194*117

p        =  dim(X)[2]

sum      =  as.vector(apply(X, 2, 'sum'))
index.0  =  which(sum==0) #on these dates, new cases were 0 for all counties

mu       =   as.vector(apply(X, 2, 'mean'))
sd       =   as.vector(apply(X, 2, 'sd'))
X.orig   =   X
for (i in 1:n){
    X[i,-index.0]   =    ((X[i,-index.0]) - (mu[-index.0]))/(sd[-index.0])
}

# apply(X, 2, 'mean')
# apply(X, 2, 'sd')
#X=X.orig


n.train        =     floor(0.8*n)
n.test         =     n-n.train

M=100
Rsq.test.rf     =     rep(0,M)  # rf= randomForest
Rsq.train.rf    =     rep(0,M)
Rsq.test.en     =     rep(0,M)  # en = elastic net
Rsq.train.en    =     rep(0,M)
Rsq.test.rid    =     rep(0,M)  # rid = Ridge
Rsq.train.rid   =     rep(0,M)
Rsq.test.lasso  =     rep(0,M)  # lasso
Rsq.train.lasso =     rep(0,M)

for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  
  # ridge  alpha = 0 
  cv.fit.rid        =     cv.glmnet(X.train, y.train, alpha = 0,family = "gaussian",nfolds=10,type.measure = "mae")
  fit.rid           =     glmnet(X.train, y.train, alpha = 0, family = "gaussian", lambda = cv.fit.rid$lambda.min)
  y.train.hat       =     predict(fit.rid, newx = X.train, type = "response",cv.fit.rid$lambda.min) # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat        =     predict(fit.rid, newx = X.test, type = "response",cv.fit.rid$lambda.min) # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.rid[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2) #1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rid[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  

  
  # elastic-net    alpha = 0.5
  cv.fit.en         =     cv.glmnet(X.train, y.train, family = "gaussian", alpha = 0.5, nfolds = 10,type.measure = "mae")
  fit.en            =     glmnet(X.train, y.train,alpha = 0.5, lambda = cv.fit.en$lambda.min,family = "gaussian")
  y.train.hat       =     predict(fit.en, newx = X.train, type = "response",cv.fit.en$lambda.min) # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat        =     predict(fit.en, newx = X.test, type = "response",cv.fit.en$lambda.min) # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]    =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.en[m]   =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  

  
  # lasso    alpha = 1
  cv.fit.lasso       =     cv.glmnet(X.train, y.train, family = "gaussian", alpha = 1, nfolds = 10,type.measure = "mae")
  fit.lasso          =     glmnet(X.train, y.train,alpha = 1, lambda = cv.fit.lasso$lambda.min,family = "gaussian")
  y.train.hat        =     predict(fit.lasso, newx = X.train, type = "response",cv.fit.lasso$lambda.min) # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat         =     predict(fit.lasso, newx = X.test, type = "response",cv.fit.lasso$lambda.min) # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.lasso[m]  =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.lasso[m] =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  

  
  # fit RF and calculate and record the train and test R squares 
  rf               =     randomForest(X.train, y.train, mtry = round(sqrt(p)), importance = TRUE)
  y.test.hat       =     predict(rf, X.test)
  y.train.hat      =     predict(rf, X.train)
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  

  
  cat(sprintf("m=%3.f| Rsq.test.rf=%.2f,  Rsq.test.en=%.2f| Rsq.train.rf=%.2f,  Rsq.train.en=%.2f| \n", m,  Rsq.test.rf[m], Rsq.test.en[m],  Rsq.train.rf[m], Rsq.train.en[m]))
  cat(sprintf("m=%3.f| Rsq.test.rid=%.2f,  Rsq.test.lasso=%.2f| Rsq.train.rid=%.2f,  Rsq.train.lasso=%.2f| \n", m,  Rsq.test.rid[m], Rsq.test.lasso[m],  Rsq.train.rid[m], Rsq.train.lasso[m]))
  
}



#Rsq Plot
Rsq.train           =     data.frame(c(rep("rf", M),  rep("EN", M),  rep("ridge", M), rep("lasso",M)) , 
                                     c(Rsq.train.rf, Rsq.train.en, Rsq.train.rid, Rsq.train.lasso))
colnames(Rsq.train) =     c("method","Rsq")
Rsq.test            =     data.frame(c(rep("rf", M),  rep("EN", M),  rep("ridge", M), rep("lasso",M)), 
                                     c(Rsq.test.rf, Rsq.test.en, Rsq.test.rid, Rsq.test.lasso) )
colnames(Rsq.test)  =     c("method","Rsq")


p1 = ggplot(Rsq.train)   +     aes(x=method, y = Rsq, fill=method) +   geom_boxplot() + xlab(expression("method")) + ylab(expression("R^2")) +
  theme(legend.text = element_text(colour = "black", size = 10, face = "bold", family = "Courier")) +
  ggtitle("train") +
  theme( axis.title.x = element_text(size = 16, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 20, family   = "Courier"), 
         axis.title.y        = element_text(size = 16, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+
  ylim(0.25, 1)  

p2 = ggplot(Rsq.test)   +     aes(x=method, y = Rsq, fill=method) +   geom_boxplot() + xlab(expression("method")) + ylab(expression("R^2")) +
  theme(legend.text = element_text(colour = "black", size = 10, face = "bold", family = "Courier")) +
  ggtitle("test") +
  theme( axis.title.x = element_text(size = 16, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 20, family   = "Courier"), 
         axis.title.y        = element_text(size = 16, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+
  ylim(0.25, 1)  


grid.arrange(p1, p2, ncol=2)




#CV plots and Residual plots
cv.fit.rid          =     cv.glmnet(X.train, y.train, alpha = 0,family = "gaussian",nfolds=10,type.measure = "mae")
fit.rid             =     glmnet(X.train, y.train, alpha = 0, family = "gaussian", lambda = cv.fit.rid$lambda.min)
y.train.hat         =     predict(fit.rid, newx = X.train, type = "response",cv.fit.rid$lambda.min) # y.train.hat=X.train %*% fit$beta + fit$a0
y.test.hat          =     predict(fit.rid, newx = X.test, type = "response",cv.fit.rid$lambda.min) # y.test.hat=X.test %*% fit$beta  + fit$a0
res.rid             =     data.frame(c(rep("train", n.train),rep("test", n.test)), c(1:n), c(y.train.hat - y.train, y.test.hat - y.test))
colnames(res.rid)   =     c("Ridge", "time", "residual")
res.rid.barplot     =     ggplot(res.rid, aes(x=Ridge, y=residual)) + geom_boxplot(outlier.size = 0.5)
res.rid.barplot
plot(cv.fit.rid)


cv.fit.en           =     cv.glmnet(X.train, y.train, family = "gaussian", alpha = 0.5, nfolds = 10,type.measure = "mae")
fit.en              =     glmnet(X.train, y.train,alpha = 0.5, lambda = cv.fit.en$lambda.min,family = "gaussian")
y.train.hat         =     predict(fit.en, newx = X.train, type = "response",cv.fit.en$lambda.min) # y.train.hat=X.train %*% fit$beta + fit$a0
y.test.hat          =     predict(fit.en, newx = X.test, type = "response",cv.fit.en$lambda.min) # y.test.hat=X.test %*% fit$beta  + fit$a0
res.en              =     data.frame(c(rep("train", n.train),rep("test", n.test)), c(1:n), c(y.train.hat - y.train, y.test.hat - y.test))
colnames(res.en)    =     c("EN", "time", "residual")
res.en.barplot      =     ggplot(res.en, aes(x=EN, y=residual)) + geom_boxplot(outlier.size = 0.5)
res.en.barplot
plot(cv.fit.en)


cv.fit.lasso         =     cv.glmnet(X.train, y.train, family = "gaussian", alpha = 1, nfolds = 10,type.measure = "mae")
fit.lasso            =     glmnet(X.train, y.train,alpha = 1, lambda = cv.fit.lasso$lambda.min,family = "gaussian")
y.train.hat          =     predict(fit.lasso, newx = X.train, type = "response",cv.fit.lasso$lambda.min) # y.train.hat=X.train %*% fit$beta + fit$a0
y.test.hat           =     predict(fit.lasso, newx = X.test, type = "response",cv.fit.lasso$lambda.min) # y.test.hat=X.test %*% fit$beta  + fit$a0
res.lasso            =     data.frame(c(rep("train", n.train),rep("test", n.test)), c(1:n), c(y.train.hat - y.train, y.test.hat - y.test))
colnames(res.lasso)  =     c("lasso", "time", "residual")
res.lasso.barplot    =     ggplot(res.lasso, aes(x=lasso, y=residual)) + geom_boxplot(outlier.size = 0.5)
res.lasso.barplot
plot(cv.fit.lasso)


rf                =     randomForest(X.train, y.train, mtry = round(sqrt(p)), importance = TRUE)
y.test.hat        =     predict(rf, X.test)
y.train.hat       =     predict(rf, X.train)
res.rf            =     data.frame(c(rep("train", n.train),rep("test", n.test)), c(1:n),c(y.train.hat - y.train, y.test.hat - y.test))
colnames(res.rf)  =     c("rf", "time", "residual")
res.rf.barplot    =     ggplot(res.rf, aes(x=rf, y=residual)) + geom_boxplot(outlier.size = 0.5)
res.rf.barplot



#Bootstrap

bootstrapSamples =     100
beta.rf.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.en.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)
beta.rid.bs      =     matrix(0, nrow = p, ncol = bootstrapSamples)         
beta.lasso.bs    =     matrix(0, nrow = p, ncol = bootstrapSamples)         

for (m in 1:bootstrapSamples){
  bs_indexes     =     sample(n, replace=T)
  X.bs           =     X[bs_indexes, ]
  y.bs           =     y[bs_indexes]

# fit bs rf
rf               =     randomForest(X.bs, y.bs, mtry = round(sqrt(p)), importance = TRUE)
beta.rf.bs[,m]   =     as.vector(rf$importance[,1])

# fit bs en
cv.fit.en        =     cv.glmnet(X.bs, y.bs, alpha = 0.5, nfolds = 10, family = "gaussian", type.measure = "mae")
fit.en           =     glmnet(X.bs, y.bs, alpha = 0.5, lambda = cv.fit.en$lambda.min, family = "gaussian")  
beta.en.bs[,m]   =     as.vector(fit.en$beta)

# fit bs ridge
cv.fit.rid       =     cv.glmnet(X.bs, y.bs, alpha = 0, nfolds = 10,  family = "gaussian", type.measure = "mae")
fit.rid          =     glmnet(X.bs, y.bs, alpha = 0, lambda = cv.fit.rid$lambda.min,  family = "gaussian")  
beta.rid.bs[,m]  =     as.vector(fit.rid$beta)

# fit bs lasso
cv.fit.lasso     =     cv.glmnet(X.bs, y.bs, alpha = 1, nfolds = 10,  family = "gaussian", type.measure = "mae")
fit.lasso        =     glmnet(X.bs, y.bs, alpha = 1, lambda = cv.fit.lasso$lambda.min,  family = "gaussian")  
beta.lasso.bs[,m]=     as.vector(fit.lasso$beta)
cat(sprintf("Bootstrap Sample %3.f \n", m))
}



# calculate bootstrapped standard errors / alternatively you could use qunatiles to find upper and lower bounds
rf.bs.sd    = apply(beta.rf.bs, 1, "sd")
en.bs.sd    = apply(beta.en.bs, 1, "sd")
rid.bs.sd   = apply(beta.rid.bs, 1, "sd")
lasso.bs.sd = apply(beta.lasso.bs, 1, "sd")


# fit rf to the whole data
rf               =     randomForest(X, y, mtry = round(sqrt(p)), importance = TRUE)

# fit en to the whole data
cv.fit.en        =     cv.glmnet(X, y, alpha = 0.5, nfolds = 10,family = "gaussian", type.measure = "mae")
fit.en           =     glmnet(X, y, alpha = 0.5, lambda = cv.fit.en$lambda.min,family = "gaussian")

# fit ridge to the whole data
cv.fit.rid       =     cv.glmnet(X, y, alpha = 0, nfolds = 10, family = "gaussian", type.measure = "mae")
fit.rid          =     glmnet(X, y, alpha = 0, lambda = cv.fit.rid$lambda.min, family = "gaussian")

# fit lasso to the whole data
cv.fit.lasso           =     cv.glmnet(X, y, alpha = 1, nfolds = 10,family = "gaussian", type.measure = "mae")
fit.lasso              =     glmnet(X, y, alpha = 1, lambda = cv.fit.lasso$lambda.min, family = "gaussian")


betaS.rf               =     data.frame(c(1:p), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "RFfeature", "value", "err")

betaS.en               =     data.frame(c(1:p), as.vector(fit.en$beta), 2*en.bs.sd)
colnames(betaS.en)     =     c( "ENfeature", "value", "err")

betaS.rid              =     data.frame(c(1:p), as.vector(fit.rid$beta), 2*rid.bs.sd)
colnames(betaS.rid)    =     c( "RIDGEfeature", "value", "err")

betaS.lasso            =     data.frame(c(1:p), as.vector(fit.lasso$beta), 2*lasso.bs.sd)
colnames(betaS.lasso)  =     c( "LASSOfeature", "value", "err")



rfPlot =  ggplot(betaS.rf, aes(x=RFfeature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) 

enPlot =  ggplot(betaS.en, aes(x=ENfeature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

ridPlot =  ggplot(betaS.rid, aes(x=RIDGEfeature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

lassoPlot =  ggplot(betaS.lasso, aes(x=LASSOfeature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

grid.arrange(rfPlot, enPlot, ridPlot, lassoPlot,nrow = 4)







# we need to change the order of factor levels by specifying the order explicitly.
betaS.rf$RFfeature        =  factor(betaS.rf$RFfeature, levels = betaS.rf$RFfeature[order(betaS.rf$value, decreasing = TRUE)])
betaS.en$ENfeature        =  factor(betaS.en$ENfeature, levels = betaS.rf$RFfeature[order(betaS.rf$value, decreasing = TRUE)])
betaS.rid$RIDGEfeature    =  factor(betaS.rid$RIDGEfeature, levels = betaS.rf$RFfeature[order(betaS.rf$value, decreasing = TRUE)])
betaS.lasso$LASSOfeature  =  factor(betaS.lasso$LASSOfeature, levels = betaS.rf$RFfeature[order(betaS.rf$value, decreasing = TRUE)])


rfPlot =  ggplot(betaS.rf, aes(x=RFfeature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

enPlot =  ggplot(betaS.en, aes(x=ENfeature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

ridPlot =  ggplot(betaS.rid, aes(x=RIDGEfeature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

lassoPlot =  ggplot(betaS.lasso, aes(x=LASSOfeature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)


grid.arrange(rfPlot, enPlot,ridPlot, lassoPlot,nrow = 4)











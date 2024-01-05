setwd('C:/Users/default.DESKTOP-LFLCLIU/Desktop')
getwd()

install.packages("factoextra")
install.packages('arules')
install.packages('randomForest')
install.packages('ipred')
install.packages('gbm')
install.packages('xgboost')
install.packages('adabag')
install.packages('rpart')
install.packages('rpart.plot')
install.packages('ggplot2')
install.packages('data.table')
install.packages('dbscan')

library(arules)
library(randomForest) ##random Forest
library(ipred) ##bagging
library(gbm) ## Boosting
library(xgboost)  ## xgboost  
library(adabag)  ##Adaboosting : boosting
library(rpart)
library(rpart.plot)
library(ggplot2)
library(data.table)
library(factoextra)
library(cluster)
library(gridExtra)
library(dbscan)  #dbscan, knndist


########################################################################################################
##################(1) 연관규칙 분석
########################################################################################################

##########(1-a) ‘Income’ data 불러오기
install.packages("arules")
library(arules)
data(Income)


##########(1-b) 'Income' data 변수 확인
Income
summary(Income) 
colnames(Income)
inspect(Income[1,])
##6876개의 데이터가 있고 50개의 아이템이 있다.
##아이템에는 income, sex, marital status, age 등이 있다.
##아이템 중 모국어가 영어인 경우가 6277로 가장 많았다


##########(1-c) 고소득자(income=“$40,000+”) 그룹에 대한 itemFrequencyPlot를 그리고 설명하여라.
Income.H<-Income[Income %in% "income=$40,000+"]
itemFrequencyPlot(Income.H,main="Income=$40,000's ItemFrequencyPlot",topN=20)
##고속득자 그룹에 대해 itemFrequencyPlot를 보니 해당 그룹이 가장 많이 가지고 있는 아이템은 language in home=english이다.
##다음으로 ethnic classification=white, type of home=house 순으로 많았다.
##즉 income이 $40,000 이상인 그룹의 신뢰도는 language in home=english이 가장 높다고 할 수 있다.


##########(1-d) 연관규칙분석 : (rhs) 고소득자에 대한 연관규칙을 신뢰도 기준 상위 5개 추출하고 설명하
#여라. (단 최소지지도 0.1, 최소신뢰도 0.8, 향상도 1.0 적용)
ap_rules<-apriori(Income,parameter = list(support=0.1, confidence=0.8),appearance = list(rhs = "income=$40,000+"))
ap_rules.sub<-subset(ap_rules,subset= lift>1.0)

inspect(head(sort(ap_rules.sub,by= "confidence"),5))
summary(ap_rules.sub)
##연관규칙 분석 결과 (rhs)고속득자에 대한 30개의 규칙을 찾았다.그 규칙 중 가장 큰 size는 6개의 아이템을 더한 것이고 가장 작은 size는 3개의 아이템을 더한 것이다.
##신뢰도가 가장 높은 규칙은 marital status=married, occupation=professional/managerial,householder status=own 인 규칙이고
##해당 규칙의 support는 0.1042757 / confidence는 0.8754579, lift는 2.318817로 나타났다.
##즉 marital status=married, occupation=professional/managerial,householder status=own이면 income이 $40000이상이라는 결과가 나온다.


########################################################################################################
##########(2)계층적 군집분석
########################################################################################################

##########(2-a)
df<-USArrests
D1<-dist(df)
round(D1,2)

hc1<-hclust(D1^2,method='complete')
plot(hc1, labels=row.names(df), hang=0.1, main="dendrogram : 최장 연결법")

hc1$height ##dendrogram의 높이는 5.25부터 86214.31까지 있다.


##########(2-b)
c1.num <- 3  #군집 수 설정
hc1.result <- cutree(hc1, k = c1.num) #최장 연결법 결과
hc1.result

##군집 1에는 Alabama, Alaska,  Arizona, California, Delaware, Florida, Illinois, Louisiana, Maryland, Michigan, Mississippi, Nevada, New Mexico, New York, North Carolina, South Carolina 
##군집 2에는 Arkansas, Colorado, Georgia, Massachusetts,  Missouri, New Jersey, Oklahoma, Oregon, Rhode Island,  Tennessee, Texas, Virginia, Washington, Wyoming
##군집 3에는 Connecticut, Hawaii, Idaho, Indiana, Iowa, Kansas, Kentucky, Maine, Minnesota, Montana, Nebraska, New Hampshire, North Dakota, Ohio, Pennsylvania, South Dakota, Utah, Vermont, West Virginia, Wisconsin

plot(hc1, labels=row.names(df), hang=0.1, main="dendrogram : 최장 연결법")
abline(h=17000,col="red",lty=3)


##########(2-c)
sc_df <- scale(df)  ## 표준화 

hc2_sc <- hclust(dist(sc_df)^2, method = "complete")   #최장 연결법
hc2_sc$height
plot(hc2_sc, labels = row.names(df), hang = 0.1, main = "dendrogram : 최장 연결법, 표준화")


##########(2-d)
par(mfrow=c(1,2))
plot(hc1, labels=row.names(df), hang=0.1, main="dendrogram : 최장 연결법")
plot(hc2_sc, labels = row.names(df), hang = 0.1, main = "dendrogram : 최장 연결법, 표준화")

##dendrogram을 보면 표준화를 한 후 최장 연결법으로 tree를 그린 경우 군집에 4개로 이루어지는 것을 확인할 수 있다.
##이는 군집분석에서는 거리를 사용할 경우 변수의 단위에 의해 영향을 받기 때문이다.


########################################################################################################
##########(3)분류분석
########################################################################################################

##########(3-a) 
data<-read.csv('sample_DT.csv',header=T,stringsAsFactors=T)
head(data)
as.factor(data$DEFECT_TYPE)

set.seed(1234)
train_id<-sample(1:nrow(data),nrow(data)*0.7)
train_dt<-data[train_id,]
test_dt<-data[-train_id,]
nrow(train_dt)
nrow(test_dt)


##########(3-b) logistics
glm.fit=glm(DEFECT_TYPE ~ ., data=train_dt, family=binomial)
summary(glm.fit)

p <- ifelse(predict(glm.fit, test_dt, type = "response") > 0.3, "NG", "G")

table(test_dt$DEFECT_TYPE, p)
mean(test_dt$DEFECT_TYPE != p) ##test data의 적합 결과 오분률이 22.9%가 나왔다.


p <- ifelse(predict(glm.fit, test_dt, type = "response") > 0.5, "NG", "G")

table(test_dt$DEFECT_TYPE, p)
mean(test_dt$DEFECT_TYPE != p) ##test data의 적합 결과 오분률이 19%가 나왔다.

p <- ifelse(predict(glm.fit, test_dt, type = "response") > 0.7, "NG", "G")

table(test_dt$DEFECT_TYPE, p)
mean(test_dt$DEFECT_TYPE != p) ##test data의 적합 결과 오분률이 21%가 나왔다.


##########(3-c) 의사결정나무(가지치기 실행)
set.seed(1234)
tree_data <- rpart(DEFECT_TYPE ~ ., data = train_dt,
                   control = rpart.control(cp = 0.0001,
                                           minsplit = 10))

tree_data

summary(tree_data)
rpart.plot(tree_data)

yhat = predict(tree_data, newdata = test_dt, type = "class")

mean(yhat!=test_dt$DEFECT_TYPE) ##test data의 적합 결과 오분률이 20.4%가 나왔다.

printcp(tree_data)
plotcp(tree_data) ##xerror가 0.42317이 가장 작으므로 새로운 cp는 0.42317+0.011131=약 0.433에 근거하여 정한다.
##따라서 새로운 cp는 0.003로 정한다.


#가지치기
prune_tree_data = rpart(DEFECT_TYPE ~ ., data = train_dt,
                        control = rpart.control(cp = 0.003))

rpart.plot(prune_tree_data)

yhat = predict(prune_tree_data, newdata = test_dt, type = "class")

table(test_dt$DEFECT_TYPE, yhat)

mean(test_dt$DEFECT_TYPE != yhat) ##가지치기 후 test data의 적합 결과 오분률이 18.7%가 나왔다.


##########(3-d) bagging
fit.bagg<- ipredbagg(train_dt[,21], ##예측하고자 하는 y값
                     train_dt[,-21], ##DEFECT_TYPE를 제외한 x값
                     nbagg=500, ##트리를 500개 만들겠다.
                     coob=T) 

fit.bagg

pred<-predict(fit.bagg,newdata = test_dt)
mean(test_dt$DEFECT_TYPE!=pred) ##test data의 적합 결과 오분률이 13.3%가 나왔다.




##########(3-e) boosting
data2<-read.csv('sample_DT.csv',header=T)
data2[,'DEFECT_TYPE']=as.numeric(data2[,'DEFECT_TYPE']=='G')

train_dt2<-data2[train_id,]
test_dt2<-data2[-train_id,]

boosting_data<- gbm(DEFECT_TYPE~.,
                    data=train_dt2,
                    distribution="bernoulli",
                    interaction.depth = 1,
                    n.trees=500)
boosting_data ##20 had non-zero influence.로 보아 20개 변수가 모두 중요함을 알 수 있다.
summary(boosting_data) ##valueG가 가장 중요한 변수임을 알 수 있다.

yhat.boost <- predict(boosting_data,
                      newdata = test_dt2, n.trees = 500,type='response')

yhat.boost1<-ifelse(yhat.boost>0.5,1,0)
mean(test_dt2$DEFECT_TYPE!=yhat.boost1) ##test data의 적합 결과 오분률이 15.3%가 나왔다.

yhat.boost1<-ifelse(yhat.boost>0.6,1,0)
mean(test_dt2$DEFECT_TYPE!=yhat.boost1) ##test data의 적합 결과 오분률이 15.27%가 나왔다.

yhat.boost1<-ifelse(yhat.boost>0.7,1,0)
mean(test_dt2$DEFECT_TYPE!=yhat.boost1) ##test data의 적합 결과 오분률이 18.3%가 나왔다.


##number of interaction.depth
Boost_ID <- function(d){
  
  tmp_boost_model <- gbm(DEFECT_TYPE~.,data=train_dt2,
                         distribution="bernoulli",
                         interaction.depth = d,
                         n.trees=100)
  
  return(tmp_boost_model$oobag.improve)
  
}

tmp_dt <- data.table( num_tree = 1:100,
                      Boost_1 = Boost_ID(1),
                      Boost_2 = Boost_ID(2),
                      Boost_4 = Boost_ID(4),
                      Boost_12 = Boost_ID(12)
)

melt.tmp <- melt(tmp_dt, id=1)

ggplot(melt.tmp, aes(num_tree, value, col=variable)) + 
  geom_line(lwd=2) +
  labs(y='oobag.improve', col="depth")+ theme_bw()  ##depth의 차이는 크게 없다.


##########(3-f) randomForest
set.seed(1234)
rf.data <- randomForest(DEFECT_TYPE ~ ., 
                        data = train_dt,
                        ntree=500,
                        mtry = 6, ##설명변수를 6개로 설정한다.
                        importance = TRUE)  ##변수 중요도를 출력한다

rf.data ##train_dt의 오분율은 14.97%이다.

yhat.bag <- predict(rf.data, newdata =test_dt )

mean(test_dt$DEFECT_TYPE!=yhat.bag) ##test data의 적합 결과 오분률이 13.4%가 나왔다.

rf.data$importance ##Gini계수 감소량이 valueG가 가장 높아 valueG가 중요한 변수이다.
varImpPlot(rf.data, pch=16)


rf.data <- randomForest(DEFECT_TYPE ~ ., 
                        data = train_dt,
                        ntree=500,
                        mtry = 4, ##설명변수를 4개로 설정한다.
                        importance = TRUE)  ##변수 중요도를 출력한다

rf.data ##train_dt의 오분율은 14.77%이다.

yhat.bag <- predict(rf.data, newdata =test_dt )

mean(test_dt$DEFECT_TYPE!=yhat.bag) #test data의 적합 결과 오분률이 13.3%가 나왔다.

rf.data <- randomForest(DEFECT_TYPE ~ ., 
                        data = train_dt,
                        ntree=500,
                        mtry = 10, ##설명변수를 10개로 설정한다.
                        importance = TRUE)  ##변수 중요도를 출력한다

rf.data ##train_dt의 오분율은 14.63%이다.

yhat.bag <- predict(rf.data, newdata =test_dt )

mean(test_dt$DEFECT_TYPE!=yhat.bag) #test data의 적합 결과 오분률이 13%가 나왔다.

rf.data <- randomForest(DEFECT_TYPE ~ ., 
                        data = train_dt,
                        ntree=500,
                        mtry = 15, ##설명변수를 20개로 설정한다.
                        importance = TRUE)  ##변수 중요도를 출력한다

rf.data ##train_dt의 오분율은 14.79%이다.

yhat.bag <- predict(rf.data, newdata =test_dt )

mean(test_dt$DEFECT_TYPE!=yhat.bag) #test data의 적합 결과 오분률이 13.2%가 나왔다.

##설명변수의 개수를 10개로 하는 것이 test error를 가장 낮게 할 수 있다.

########################################################################################################
##################(4) 군집분석
########################################################################################################
data2<-read.csv('Wholesale_customers_data.csv',header=T,stringsAsFactors=T)
data2<-data2[,3:8]

##########(4-a) 계층적 군집분석
hc2_sc <- hclust(dist(data2)^2, method = "complete")   #최장 연결법

plot(hc2_sc, labels = row.names(data2), hang = 0.1, main = "complete : 최장 연결법")
rect.hclust(hc2_sc, k= 2, border = "red")


c1.num <- 2

hc2.result <- cutree(hc2_sc, k = c1.num)

sum(hc2.result==1)
sum(hc2.result==2)

##########(4-b) k-means
data2_sc2 = data.frame(scale(data2))
set.seed(1234)
data_k <-  kmeans(data2_sc2, centers = 3, nstart = 20)

data_k$cluster
sum(data_k$cluster==1)
sum(data_k$cluster==2)
sum(data_k$cluster==3)

data_k$centers
data_k$withinss
data_k$tot.withinss
data_k$size
data_k$iter ##반복이 3번 이루어졌다.

fviz_cluster(data_k, data2)

data_k <-  kmeans(data2_sc2, centers = 2, nstart = 20)

data_k$cluster
sum(data_k$cluster==1)
sum(data_k$cluster==2)
data_k$centers
data_k$withinss
data_k$tot.withinss
data_k$size
data_k$iter ##반복이 3번 이루어졌다.

fviz_cluster(data_k, data2)

##########(4-c) pam
fit = pam(data2_sc2,k=3)

summary(fit)

fit$medoids #Medoid
fit$id.med # Medoid samples 
fit$clusinfo # cluster summary
fit$clustering # cluster index 

sum(fit$clustering==1)
sum(fit$clustering==2)
sum(fit$clustering==3)

fviz_cluster(fit,data2_sc2)

# validation of clustering: fviz_nbclust
wss = fviz_nbclust(data2_sc2, pam, method = "wss")
p1 <- wss+theme(axis.text = element_text(size = 8, color = "red"), 
                title = element_text(size = 8, color = "blue"))
wss$data

sil = fviz_nbclust(data2_sc2, pam, method = "silhouette")
p2 <- sil+theme(axis.text = element_text(size = 8, color = "red"), 
                title = element_text(size = 8, color = "blue"))
sil$data

grid.arrange(p1, p2, nrow = 1)

wss = fviz_nbclust(data2_sc2, kmeans, method = "wss")
p2 <- wss+theme(axis.text = element_text(size = 8, color = "red"), 
                title = element_text(size = 8, color = "blue"))
wss$data

sil = fviz_nbclust(data2_sc2, kmeans, method = "silhouette")
p2 <- sil+theme(axis.text = element_text(size = 8, color = "red"), 
                title = element_text(size = 8, color = "blue"))
sil$data

grid.arrange(p1, p2, nrow = 1)
##Elbow method로 k의 개수가 결정하기 위해 그래프를 그려보았다. 그 결과 k의 개수가 8에서 9로 넘어갈 때
##군집내 응집도 변화량이 가장 작으므로 적절한 k의 수를 2로 결정하였다.

fit = pam(data2_sc2,k=2)

summary(fit)

fit$medoids #Medoid
fit$id.med # Medoid samples 
fit$clusinfo # cluster summary
fit$clustering # cluster index 

sum(fit$clustering==1)
sum(fit$clustering==2)

fviz_cluster(fit,data2_sc2)

####################################################
##########(4-d) DBscan
####################################################
kNNdistplot(data2, k = 5)
kNNdist(data2, k = 5)
abline(h=16000,col='red')
##5개 이상 있으면 하나의 군집으로 인식하겠다.
##plot을 확인한 결과 eps가 10000이면 대부분의 값이 군집에 속할 수 있다.

eps <- 16000
res <- dbscan(data2, eps = eps , minPts = 5)
str(res)
res

fviz_cluster(res,data2_sc2)

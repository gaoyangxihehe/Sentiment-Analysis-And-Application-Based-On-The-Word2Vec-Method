library("tm")
library("tsne")
library("pROC")
library("e1071")
library("glmnet")
library("jiebaR")
library("stringr")
library("wordcloud2")
library("wordVectors")
#读取数据
source_train <- DirSource("train", encoding = "UTF-8")
d.corpus_train <- Corpus(source_train, readerControl = list(language = NA))
head(as.character(d.corpus_train[[1]]))

source_test <- DirSource("test", encoding = "UTF-8")
d.corpus_test <- Corpus(source_test, readerControl = list(language = NA))
head(as.character(d.corpus_test[[1]]))
#清洗数据
d.corpus_train <- tm_map(d.corpus_train, removePunctuation)         #移除标点符号
d.corpus_train <- tm_map(d.corpus_train, removeNumbers)             #移除数字
d.corpus_train <- tm_map(d.corpus_train, str_replace_all,
                   pattern = "[a-zA-Z]+", replacement = "")         #移除英文字母
d.corpus_train <- tm_map(d.corpus_train, stripWhitespace)           #移除多余的空格

d.corpus_test <- tm_map(d.corpus_test, removePunctuation)           #移除标点符号
d.corpus_test <- tm_map(d.corpus_test, removeNumbers)               #移除数字
d.corpus_test <- tm_map(d.corpus_test, str_replace_all,
                   pattern = "[a-zA-Z]+", replacement = "")         #移除英文字母
d.corpus_test <- tm_map(d.corpus_test, stripWhitespace)             #移除多余的空格
#分词
mix <- worker(stop_word = "stop_words.utf8")                        #读入停止词词典
mix$bylines = TRUE                                                  #设置为分行输出
d.corpus_train <- tm_map(d.corpus_train, segment, mix)              #分词
inspect(d.corpus_train[7147])                                       #177 1250 7149 为空

d.corpus_test <- tm_map(d.corpus_test, segment, mix)                #分词
#找出空值
a <- vector(length = length(d.corpus_train))
for (i in 1:length(d.corpus_train)) {
    a[i] <- length(d.corpus_train[[i]]$content)
}
b <- ifelse(a == 0, 1, 0)
which(b == 1)
#删除空值
d.corpus_train <- d.corpus_train[-177]
d.corpus_train <- d.corpus_train[-1249]
d.corpus_train <- d.corpus_train[-7147]
length(d.corpus_train)
#构造DTM
train_tf <- DocumentTermMatrix(d.corpus_train)                      #TF
train_tfidf <- DocumentTermMatrix(d.corpus_train, 
                                control = list(weighting = weightTfIdf))
test_tf <- DocumentTermMatrix(d.corpus_test)                        #TF
test_tfidf <- DocumentTermMatrix(d.corpus_test, 
                                 control = list(weighting = weightTfIdf))
####vec对应的doc向量
q <- findFreqTerms(train_tf, 5)
vec_train_tfidf <- train_tfidf[, q]
w <- intersect(q, colnames(test_tfidf))
vec_test_tfidf <- test_tfidf[, w]

vec_train_tfidf <- as.matrix(vec_train_tfidf)
vec_test_tfidf <- as.matrix(vec_test_tfidf)
#挑选变量
a <- findFreqTerms(train_tf, 100)
train_tf <- train_tf[, a]
train_tfidf <- train_tfidf[, a]
test_tf <- test_tf[, a]
test_tfidf <- test_tfidf[, a]

train_tf <- as.matrix(train_tf)
train_tfidf <- as.matrix(train_tfidf)
test_tf <- as.matrix(test_tf)
test_tfidf <- as.matrix(test_tfidf)

#绘制词云图
train_freq <- colSums(train_tf)
train_freq <- sort(train_freq, decreasing = TRUE)                   #对词频按降序排列
train_words <- names(train_freq)
train_word_freq <- data.frame(words = train_words, freq = train_freq)           #构建word-freq数据框
wordcloud2(train_word_freq[1:100, ], size = 0.7, color = "random-light", 
           backgroundColor = "white", shape = "oval")
#卡方检验
chi_tf <- ifelse(train_tf == 0, 1, 0)
neg_tf <- colSums(chi_tf[1:2398, ])
pos_tf <- colSums(chi_tf[2399:7997, ])
chi_mat <- rbind(2398 - neg_tf, neg_tf, 5599 - pos_tf, pos_tf)
chi <- apply(chi_mat, 2, function(x) 
    chisq.test(matrix(x, nrow = 2))$statistic)
names(chi) <- colnames(train_tf)
#特征选取
chi <- sort(chi, decreasing = T)
chi <- chi[1:sum(chi > 3.84)]                           #选取95%置信区间
length(chi)
train <- train_tfidf[, names(chi)]
test <- test_tfidf[, names(chi)]
#删除空值
e <- rowSums(train)
train <- train[ifelse(e == 0, FALSE, TRUE), ]
d <- rowSums(test)
test <- test[ifelse(d == 0, FALSE, TRUE), ]
#构造训练集、测试集
train_re <- rep(c(0,1), c(2390, 5584))
test_re <- rep(c(0,1), c(600, 1394))
train <- data.frame(cbind(train_re, scale(train)))
test <- data.frame(cbind(test_re, scale(test)))
train$train_re <- as.factor(train$train_re)
test$test_re <- as.factor(test$test_re)
###################################################### TFIDF准备好数据啦！######
#logistic回归
model1 <- glm(train_re ~ ., data = train, family = binomial(link = "logit"))
summary(model1)
model1_prob <- predict(model1, newdata = test, type = "response")
#绘制ROC曲线
model1_roc <- roc(test$test_re, model1_prob)
plot(model1_roc, print.auc = TRUE, auc.polygon = TRUE, grid = c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE)
model1_pre <- rep(1, 1994)
model1_pre[model1_prob < 0.657] <- 0
cft <- table(model1_pre, test$test_re)
require(caret)
confusionMatrix(cft, positive = "1")
#准确率0.839 精确率0.9182 召回率0.8451  F1 0.88 AUC 0.9

#LASSO
fit <- glmnet(as.matrix(train[, -1]), train$train_re, family = "binomial",
              nlambda = 50, alpha = 1)
plot(fit, xvar = "lambda", label=TRUE)
cvfit <- cv.glmnet(as.matrix(train[, -1]), train$train_re, family = "binomial",  
                   type.measure="class", nfolds = 5)
plot(cvfit)
model2_prob <- predict(cvfit, newx = as.matrix(test[, -1]), s = "lambda.1se",
                       type = "response")
xishu <- colnames(test)[coef(cvfit)@i[-1] +1]
length(xishu)
#绘制ROC曲线
model2_roc <- roc(test$test_re, model2_prob)
plot(model2_roc, print.auc = TRUE, auc.polygon = TRUE, grid = c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE)
model2_pre <- rep(1, 1994)
model2_pre[model2_prob < 0.611] <- 0
cft2 <- table(model2_pre, test$test_re)
confusionMatrix(cft2, positive = "1")
#准确率0.8516  精确率0.9147 召回率0.8687  F1 0.8911068 AUC 0.903
#NaiveBayes
nb <- naiveBayes(train_re ~ ., data = train)
nb_prob <- predict(nb, test, type = "raw")
#绘制ROC曲线
model3_roc <- roc(test$test_re, nb_prob[, 2])
plot(model3_roc, print.auc = TRUE, auc.polygon = TRUE, grid = c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE)
nb_pre <- rep(1, 1994)
nb_pre[nb_prob[, 2] < 0.788] <- 0
cft3 <- table(nb_pre, test$test_re)
confusionMatrix(cft3, positive = "1")
#准确率0.7367   精确率0.9027 召回率0.6987  F1 0.787 AUC 0.813
###############################################################################
#Word2Vec
#生成数据
for (i in 1:7997) {
    write.table(c(d.corpus_train[[i]]$content, "\n"), file = "data.txt", 
                append = T, quote = F, col.name = F, row.names = F, eol = " ")
}
#生成词向量
model <- train_word2vec("data.txt", "data.bin", vectors = 100, threads = 4,
                        window = 5, cbow = 1, min_count = 5, iter = 5, 
                        negative_samples = 10)
vec <- as.matrix(model@.Data[-1, ])
rownames(vec) <- rownames(model@.Data)[-1]
dim(vec)
#描述性分析
#与词相近
model %>% closest_to("不好")
model %>% closest_to("不错")
#与多个词相近
set.seed(10)
centers = 150
clustering = kmeans(model, centers = centers, iter.max = 40)
sapply(sample(1:centers, 10), function(n) {
    names(clustering$cluster[clustering$cluster == n][1:10])
})
#相加
model %>% closest_to(~ "不太好" + "安静", n = 15)
#树形图
ingredients = c("服务","火车站","房间","不好")
term_set = lapply(ingredients, 
                  function(ingredient) {
                      nearest_words = model %>% closest_to(model[[ingredient]], 20)
                      nearest_words$word
                  }) %>% unlist
subset = model[[term_set, average = F]]
subset %>%
    cosineDist(subset) %>% 
    as.dist %>%
    hclust %>%
    plot
#三类平面图
tastes = model[[c("不好","服务","房"),average=F]]
common_similarities_tastes = model[1:2000,] %>% cosineSimilarity(tastes)
high_similarities_to_tastes = common_similarities_tastes[
    rank(-apply(common_similarities_tastes,1,max)) < 55, ]

high_similarities_to_tastes %>% 
    prcomp %>% 
    biplot()
#平面图
plot(model, perplexity = 50)
#段落向量
#训练集
vec_train <- matrix(data = NA, nrow = 7997, ncol = 100)
for (i in 1:7997) {
    vec_train[i, ] <- model[[c(d.corpus_train[[i]]$content), average = T]]@.Data[, ]
}
#测试集
vec_test <- matrix(data = NA, nrow = 2000, ncol = 100)
for (i in 1:2000) {
    vec_test[i, ] <- model[[c(d.corpus_test[[i]]$content), average = T]]@.Data[, ]
}
#构造训练集、测试集
train1_re <- rep(c(0,1), c(2398, 5598))
test1_re <- rep(c(0,1), c(600, 1400))
vec_train <- vec_train[-4753, ]
train1 <- data.frame(cbind(train1_re, scale(vec_train)))
test1 <- data.frame(cbind(test1_re, scale(vec_test)))
train1$train1_re <- as.factor(train1$train1_re)
test1$test1_re <- as.factor(test1$test1_re)
############################# Word2vec 准备好数据啦！###########################
#logistic回归
vec_model1 <- glm(train1_re ~ ., data = train1, family = binomial(link = "logit"))
summary(vec_model1)
vec_model1_prob <- predict(vec_model1, newdata = test1, type = "response")
#绘制ROC曲线
vec_model1_roc <- roc(test1$test1_re, vec_model1_prob)
plot(vec_model1_roc, print.auc = TRUE, auc.polygon = TRUE, grid = c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE)
vec_model1_pre <- rep(1, 2000)
vec_model1_pre[vec_model1_prob < 0.666] <- 0
cft <- table(vec_model1_pre, test1$test1_re)
require(caret)
confusionMatrix(cft, positive = "1")
#准确率0.8595 精确率0.9255 召回率0.8693  F1 0.8965201 AUC 0.917

#LASSO
vec_fit <- glmnet(as.matrix(train1[, -1]), train1$train1_re, family = "binomial",
              nlambda = 50, alpha = 1)
plot(vec_fit, xvar = "lambda", label=TRUE)
vec_cvfit <- cv.glmnet(as.matrix(train1[, -1]), train1$train1_re, family = "binomial",  
                  type.measure="class", nfolds = 5)
plot(vec_cvfit)
vec_model2_prob <- predict(vec_cvfit, newx = as.matrix(test1[, -1]), s = "lambda.1se",
                       type = "response")
vec_xishu <- colnames(test1)[coef(vec_cvfit)@i[-1] +1]
vec_xishu
#绘制ROC曲线    
vec_model2_roc <- roc(test1$test1_re, vec_model2_prob)
plot(vec_model2_roc, print.auc = TRUE, auc.polygon = TRUE, grid = c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE)
vec_model2_pre <- rep(1, 2000)
vec_model2_pre[vec_model2_prob < 0.634] <- 0
cft2 <- table(vec_model2_pre, test1$test1_re)
confusionMatrix(cft2, positive = "1")
#准确率0.8435  精确率0.9345 召回率0.8350  F1 0.8819525 AUC 0.916
#NaiveBayes
vec_nb <- naiveBayes(train1_re ~ ., data = train1)
vec_nb_prob <- predict(vec_nb, test1, type = "raw")
#绘制ROC曲线
vec_model3_roc <- roc(test1$test1_re, vec_nb_prob[, 2])
plot(vec_model3_roc, print.auc = TRUE, auc.polygon = TRUE, grid = c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE)
vec_nb_pre <- rep(1, 2000)
vec_nb_pre[vec_nb_prob[, 2] < 0.005] <- 0
cft3 <- table(vec_nb_pre, test1$test1_re)
confusionMatrix(cft3, positive = "1")
#准确率0.8155  精确率0.8582 召回率0.8821  F1 0.8699 AUC 0.827
################################################################################
#两矩阵乘
train_name <-colnames(vec_train_tfidf)
vec_train2 <- vec_train_tfidf %*% vec[train_name, ]

test_name <-colnames(vec_test_tfidf)
vec_test2 <- vec_test_tfidf %*% vec[test_name, ]
#构造训练集、测试集
train2_re <- rep(c(0,1), c(2398, 5599))
test2_re <- rep(c(0,1), c(600, 1400))
train2 <- data.frame(cbind(train2_re, scale(vec_train2)))
test2 <- data.frame(cbind(test2_re, scale(vec_test2)))
train2$train2_re <- as.factor(train2$train2_re)
test2$test2_re <- as.factor(test2$test2_re)
##################数据准备好啦！################################################
#logistic回归
mul_model1 <- glm(train2_re ~ ., data = train2, family = binomial(link = "logit"))
summary(mul_model1)
mul_model1_prob <- predict(mul_model1, newdata = test2, type = "response")
#绘制ROC曲线
mul_model1_roc <- roc(test2$test2_re, mul_model1_prob)
plot(mul_model1_roc, print.auc = TRUE, auc.polygon = TRUE, grid = c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE)
mul_model1_pre <- rep(1, 2000)
mul_model1_pre[mul_model1_prob < 0.705] <- 0
cft <- table(mul_model1_pre, test2$test2_re)
require(caret)
confusionMatrix(cft, positive = "1")
#准确率0.8515 精确率0.9213 召回率0.8614  F1 0.8903 AUC 0.913

#LASSO
mul_fit <- glmnet(as.matrix(train2[, -1]), train2$train2_re, family = "binomial",
                  nlambda = 50, alpha = 1)
plot(mul_fit, xvar = "lambda", label=TRUE)
mul_cvfit <- cv.glmnet(as.matrix(train2[, -1]), train2$train2_re, family = "binomial",  
                       type.measure="class", nfolds = 5)
plot(mul_cvfit)
mul_model2_prob <- predict(mul_cvfit, newx = as.matrix(test2[, -1]), s = "lambda.1se",
                           type = "response")
mul_xishu <- colnames(test2)[coef(mul_cvfit)@i[-1] +1]
mul_xishu
#绘制ROC曲线
mul_model2_roc <- roc(test2$test2_re, mul_model2_prob)
plot(mul_model2_roc, print.auc = TRUE, auc.polygon = TRUE, grid = c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE)
mul_model2_pre <- rep(1, 2000)
mul_model2_pre[mul_model2_prob < 0.707] <- 0
cft2 <- table(mul_model2_pre, test2$test2_re)
confusionMatrix(cft2, positive = "1")
#准确率0.8425 精确率0.9255  召回率0.8429  F1 0.8822 AUC 0.911
#NaiveBayes
mul_nb <- naiveBayes(train2_re ~ ., data = train2)
mul_nb_prob <- predict(mul_nb, test2, type = "raw")
#绘制ROC曲线
mul_model3_roc <- roc(test2$test2_re, mul_nb_prob[, 2])
plot(mul_model3_roc, print.auc = TRUE, auc.polygon = TRUE, grid = c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE)
mul_nb_pre <- rep(1, 2000)
mul_nb_pre[mul_nb_prob[, 2] < 0.714] <- 0
cft3 <- table(mul_nb_pre, test2$test2_re)
confusionMatrix(cft3, positive = "1")
#准确率0.7915  精确率0.9099 召回率0.7793  F1 0.8349 AUC 0.857
f1 <- function(p, r){
    f <- (2 * p * r)/(p + r)
    f
}

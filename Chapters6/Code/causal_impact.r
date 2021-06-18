library(zoo)
library(BoomSpikeSlab)
library(bsts)
library(CausalImpact)
data_split<-function(data_causal,target_list,y,time)
{
  ## data_causal 为原始数据
  ## target_list 为实验区位列表 此例中为1
  ## y 为我们关心的变量名称 此例中为成交率PM2.5
  ## time 为原始数据中时间变量名称 此例中为ymd
  data_raw <- data_causal
  
  data_raw$station_id<-as.character(data_raw$station_id)
  data_raw$y<-data_raw[,which(colnames(data_raw)==paste(y))]
  data_raw$time<-data_raw[,which(colnames(data_raw)==paste(time))]
  data_raw$y<-as.numeric(as.character(data_raw$y))
  data_raw$time<-as.character(as.Date(data_raw$time))
  
  ## form the dictionary for time and index
  date_vector<-as.character(sort(unique(as.Date(data_raw$time)),decreasing = FALSE))
  ind<-seq(1:length(date_vector))
  time_ref<-data.frame(cbind(date_vector,ind))
  time_ref$date_vector<-date_vector
  time_ref$ind<-ind
  
  rm(date_vector,ind)
  
  ## form the dictionary for station_id and column names for mapping
  city_list<-sort(unique(data_raw$station_id),decreasing = FALSE)
  city_ref<-paste("station_",city_list,sep='')
  city_dic<-as.data.frame(cbind(city_list,city_ref))
  city_dic$city_list<-as.character(city_dic$city_list)
  city_dic$city_ref<-as.character(city_dic$city_ref)
  rm(city_list,city_ref)
  ## split data target and pool
  
  data_target<-data_raw[which(data_raw$station_id %in% target_list),]
  data_pool<-data_raw[-which(data_raw$station_id %in% target_list),]
  
  data_out<-list()
  data_out$data_target<-data_target
  data_out$data_pool<-data_pool
  data_out$city_dic<-city_dic
  data_out$time_ref<-time_ref
  data_out
}

pivot<-function(data_in,null_replace=NA)
{

  city_list<-sort(unique(data_in$station_id),decreasing = FALSE)
  date_vector<-as.character(sort(unique(as.Date(data_in$time)),decreasing = FALSE))
  
  data<-as.data.frame(date_vector)
  data$date_vector<-as.character(data$date_vector)
  for(i in 1:length(city_list))
  {
    temp<-data_in[which(data_in$station_id==city_list[i]),]
    for (j in 1:length(date_vector))
    {
      data[j,(i+1)]<-ifelse(
        is.null(temp[which(temp$time==date_vector[j] ),'y']),
        null_replace, 
        temp[which(temp$time==date_vector[j] ),'y'])
    }
    colnames(data)[i+1]<-paste("station_",city_list[i],sep="")
    data[,i+1]=as.numeric(data[,i+1])
  }
  missing<-vector()
  missing_rt<-vector()
  
  for(i in 1:length(city_list))
  {
    for (j in 1:length(date_vector))
    {
      missing[j]<-ifelse(is.na(data[j,(i+1)]),1,0)
    }
    missing_rt[i]<-sum(missing)/length(date_vector)
  }
  names(missing_rt)<-colnames(data)[2:length(colnames(data))]
  data_out<-list()
  data_out$pivot_data<-data
  data_out$missing_rt<-missing_rt
  data_out$time<-date_vector
  
  data_out
}

target_list<-c(1,5)


y<-"PM2.5"
time<-"ymd"

data_causal<- read.csv("~/Downloads/book_use.csv")
data_causal$station_id<-as.numeric(data_causal$station)
data_causal$ymd<-as.Date(data_causal$ymd)
data_split<-data_split(data_causal,target_list,y,time)

data_target<-data_split$data_target
data_pool<-data_split$data_pool

result_target<-pivot(data_in = data_target,null_replace=NA)
pivot_data_target<-result_target$pivot_data
missing_rate_target<-result_target$missing_rt
print(missing_rate_target)

result_pool<-pivot(data_in = data_pool,null_replace=NA)
pivot_data_pool<-result_pool$pivot_data
missing_rate_pool<-result_pool$missing_rt
print(missing_rate_pool)



## 本城市相关变量添加

inter_feature_list<-c('time','station_id','PM10','SO2','NO2','TEMP','RAIN','PRES')


data_inter_feature_raw<-data_target[,which(colnames(data_target) %in% inter_feature_list)]

for(i in 1:length(inter_feature_list)){
  if(colnames(data_inter_feature_raw)[i]=='time'||colnames(data_inter_feature_raw)[i]=='station_id'){data_inter_feature_raw[,i]<-data_inter_feature_raw[,i]}else{
    data_inter_feature_raw[,i]<-as.numeric(as.character(data_inter_feature_raw[,i]))
  }
}

names(data_inter_feature_raw)[names(data_inter_feature_raw) == "time"] <- "date_vector"
pivot_data_pool<-merge(pivot_data_pool,data_inter_feature_raw[which(data_inter_feature_raw$station_id==1),which(colnames(data_inter_feature_raw)!='station_id')],by="date_vector",all.x = TRUE)


#### 实验组数据缺失值处理 对照组数据缺失值处理

for(i in 2:dim(pivot_data_target)[2]){

  for (j in 1:dim(pivot_data_target)[1]){
    pivot_data_target[j,i]<-ifelse(is.na(pivot_data_target[j,i])==TRUE,mean(pivot_data_target[,i],na.rm = TRUE),pivot_data_target[j,i])
  }
}


for(i in 2:dim(pivot_data_pool)[2]){
  
  for (j in 1:dim(pivot_data_pool)[1]){
    pivot_data_pool[j,i]<-ifelse(is.na(pivot_data_pool[j,i])==TRUE,mean(pivot_data_pool[,i],na.rm = TRUE),pivot_data_pool[j,i])
  }
}


pivot_data_target_use<-pivot_data_target

pivot_data_pool_use<-pivot_data_pool

## 对于实验阶段添加实验项及扰动
pivot_data_target_use[which(months(as.Date(pivot_data_target_use$date_vector))=="July"),c("station_1","station_5")]<-
  pivot_data_target_use[which(months(as.Date(pivot_data_target_use$date_vector))=="July"),c("station_1","station_5")]+rnorm(n=31,mean=5,sd=3.75)


pivot_data_target[which(months(as.Date(pivot_data_target$date_vector))=="July"),c("station_1","station_5")]
pivot_data_target_use[which(months(as.Date(pivot_data_target_use$date_vector))=="July"),c("station_1","station_5")]




data_use<-cbind(y=pivot_data_target_use[,c("station_1")],pivot_data_pool[,c(2:dim(pivot_data_pool_use)[2])])

########################
data_use<-data_use[c(1:213),]
pre_period<-c(1,182)
post_period<-c(183,213)




summary(data_use)
city_default<-CausalImpact(data_use,pre.period = pre_period,post.period = post_period )
plot(city_default)
summary(city_default)

#######################
data_use<-data_use[c(1:213),]
post.period.response<-data_use$y[c(183:213)]
data_use$y[c(183:213)]<-NA

ss<-AddLocalLinearTrend(list(),data_use$y)

ss<-AddSeasonal(ss,data_use$y,nseasons = 7)
model1<-bsts(y~.,
             state.specification = ss,
             niter = 100,
             data=data_use,
             expected.model.size=1)  

model2<-bsts(y~.,
             state.specification = ss,
             niter = 100,
             data=data_use,
             expected.model.size=2)   

model3<-bsts(y~.,
             state.specification = ss,
             niter = 100,
             data=data_use,
             expected.model.size=3)     


plot(model1,"coef")     
summary(model1)
summary(model2)

CompareBstsModels(list("model1"=model1,
                       "model2"=model2,
                       "model3"=model3),
                  colors=c("black","red","blue"))

## 使用构建好的模型，分析效果
result_with_model<-CausalImpact(bsts.model = model2,post.period.response=post.period.response)    
plot(result_with_model)
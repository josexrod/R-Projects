library(tidyverse)   
library(skimr)       
library(paradox)
library(mlr3learners) 
library(mlr3measures) 
library(mlr3tuning)   
library(kknn) 
library(corrplot)
diamantes <- read.csv("diamonds.csv")
#x <- cor(diamantes, method = "pearson")
#corrplot(x, method = "number", title = "Correlacion entre variables")

diamantes <- mutate_at(diamantes,c("price"), as.numeric)
str(diamantes)
diamantes <- mutate_at(diamantes, c("cut", "color", "clarity"), as.factor)
diamantes <- subset(diamantes, select = -c(x, y))
diamantes <- diamantes[!(diamantes$z == 0),]
tarea<-TaskRegr$new(id="regresion", backend=diamantes, target="price")
regr.knn<-lrn("regr.kknn")
regr.knn$param_set
espacio_soluciones<-ParamSet$new(list(ParamInt$new("k",5,20),
                                      ParamInt$new("distance", 1, 3),
                                      ParamFct$new("kernel",c("rectangular","triangular","gaussian","inv","optimal"))))
tipo_busqueda<-mlr3tuning::tnr("random_search", batch_size = 10)
val_cruzada<-rsmp("cv",folds=10)
instancia_busqueda=TuningInstanceSingleCrit$new(task=tarea,
                                                learner = regr.knn,
                                                resampling=val_cruzada,
                                                measure = msr("regr.mse"),
                                                search_space = espacio_soluciones,
                                                terminator=trm("none"))      
future::plan("multisession")   
tipo_busqueda$optimize(instancia_busqueda)
instancia_busqueda$archive$best()
#ggplot(instancia_busqueda$archive$data, aes(x=k,y=regr.mse)) + geom_point()
#ggplot(instancia_busqueda$archive$data, aes(x=kernel,y=regr.mse)) + geom_point()
vc_externa<-rsmp("holdout", ratio=0.7) 
vc_interna<-rsmp("cv", folds=10)
optim.learner<-AutoTuner$new(learner=regr.knn,
                             vc_interna,msr("regr.mse"), 
                             espacio_soluciones,
                             terminator = trm("none"),
                             tuner=tipo_busqueda)
future::plan("multisession")
resultados<-resample(tarea,optim.learner,vc_externa)
resultados$aggregate(measures=msr("regr.mse"))
modelo_final<-optim.learner$train(tarea)
pred<-modelo_final$predict_newdata(diamantes)
resultados <- pred$response
diamantes <- cbind(diamantes, resultados)

#ggplot(diamantes, aes (x=price,y=resultados)) + geom_point()

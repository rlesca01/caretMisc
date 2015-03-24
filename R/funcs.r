


#' @title Wrapper for performing machine learning.
#'
#' @description
#' \code{CatClassML} returns models that were cross validated using specified method.
#' 
#' @param data.mod is a data.frame with pred as the last column
#' @param mods a character vector of valid models to use
#' @param mode (optional) either regress or class
#' @param parallel either TRUE or FALSE
#' @param fixedIndx TRUE or FALSE, set to true for ensemble building
#' @param grid.list a list of data.frames specifing custom grid params.  Should be made what the model is named.
#' @return A list of models equal to the number of models (mods) specified.
CatClassML <- function(data.mod, mods, mode=NULL, parallel=FALSE, method = "cv", number = 10, repeats = 3, verboseIter=TRUE, savePredictions=TRUE, tuneLength=5, fixedIndx=TRUE, grid.list=NULL){
  
  if(sum(is.na(data.mod))>0){
    warning("NAs were detected in the data.  The underlying algorithms will handle them differently.  Some will drop the whole row with an NA (gbm), others will fail, some can handle NAs without much issue (rf).")
  }
  
  if(class(data.mod)!="data.frame"){
    stop("The class of the input data must be a data frame.")  
  }
  
  if (is.null(mode)) {
    # autodetect the mode
    if (is.numeric(data.mod$pred)) {
      mode <- "regress"
    } else {
      mode <- "class"
    }
  }
  
  # set classProbs equal to TRUE if you are doing a classification problem
  if(mode=="regress"){
    classProbs=FALSE
  } else {
    classProbs=TRUE
  }
  
  model.outputs <- list()
    
  if(fixedIndx){
    # initialize parameters from the arguments that were passed
    ctrl <- trainControl(method = method, 
                         number = number, 
                         repeats = repeats, 
                         classProbs=classProbs, 
                         verboseIter=verboseIter, 
                         returnData = FALSE,
                         savePredictions=savePredictions,
                         allowParallel = parallel,
                         index=createMultiFolds(y=data.mod[,ncol(data.mod)], k=number, times=repeats))
  } else {
    # initialize parameters from the arguments that were passed
    ctrl <- trainControl(method = method, 
                         number = number, 
                         repeats = repeats, 
                         classProbs=classProbs, 
                         verboseIter=verboseIter, 
                         returnData = FALSE,
                         savePredictions=savePredictions,
                         allowParallel = parallel)
  }
  
  # run over all the rest of the algorithms
  if(length(mods)!=0){
    for(i in 1:length(mods)){
      model.outputs[[mods[i]]] <- tryCatch({train(pred~ .,
                                                data = data.mod,
                                                method = mods[i],
                                                tuneLength = tuneLength,
                                                tuneGrid = ifelse(is.data.frame(grid.list[[mods[i]]]), grid.list[[mods[i]]], NULL), # get custom grid
                                                trControl = ctrl)}, error=function(e) e)
      
    }
  }

   return(model.outputs)
}

#' @title Average predictions from repeats.
#'
#' @description
#' \code{processCVres} averages the predictions from multiple repeats together.  Should generally just be called using average param in extractCVres.
#' 
#' @param cv.res the output of extractCVres.
#' @return A matrix of cross validated, averaged, predicted values.
processCVres <- function(cv.res){
  
  ord.cv.res <- cv.res[order(cv.res$rowIndex),]
  
  f.res <- data.frame(matrix(NA, nrow=max(ord.cv.res$rowIndex), ncol=6))
  colnames(f.res) <- colnames(ord.cv.res)
  
  for(i in 1:max(ord.cv.res$rowIndex)){
    pred <- mean(ord.cv.res[ord.cv.res$rowIndex==i,]$pred, na.rm=TRUE)
    f.res[i,] <- ord.cv.res[ord.cv.res$rowIndex==i,][1,]
    f.res[i,"pred"] <- pred
  }
  
  return(f.res)
}

#' @title Nested CV performance.
#'
#' @description
#' \code{nestedCaretCVPerf} Prototype for extracting performance from nested CV.
#' 
#' @param ml.out models from caret.
#' @return Performance values.
nestedCaretCVPerf <- function(ml.out, cv.folds){
  
  suppressMessages(library(caret))
  
  rsrq <- rmse <- kappa <- accur <- matrix(NA, nrow=length(ml.out), ncol=length(ml.out[[1]]))
  
  colnames(rmse) <- colnames(rsrq) <- colnames(kappa) <- colnames(accur) <- names(ml.out[[1]])
  
  is.categ <- sum(c("Accuracy", "Kappa") %in% colnames(ml.out[[1]][[1]]$results))>0
  
  for(i in 1:length(ml.out)){
    for(j in 1:length(ml.out[[i]])){
      perf.num <- ml.out[[i]][[j]]$results
      if(is.categ){
        accur[i,j] <- max(as.numeric(as.character(perf.num$Accuracy)), na.rm = TRUE)
        kappa[i,j] <- max(as.numeric(as.character(perf.num$Kappa)), na.rm = TRUE)
      } else {
        rmse[i,j] <- min(as.numeric(as.character(perf.num$RMSE)), na.rm = TRUE)
        rsrq[i,j] <- max(as.numeric(as.character(perf.num$Rsquared)), na.rm = TRUE)
      }
    }
  }
  
  preds.oos <- matrix(NA, nrow=nrow(ds), ncol=length(ml.out[[1]]))
  colnames(preds.oos) <- names(ml.out[[1]])
  
  for(i in 1:length(ml.out)){
    for(j in 1:length(ml.out[[i]])){
      preds.oos[unlist(cv.folds[i]),j] <- predict(ml.out[[i]][[j]], newdata=ds[unlist(cv.folds[i]),])  
    }
  }
  
  oos.rsrq <- oos.rmse <- oos.acc <- oos.kappa <- vector(mode="numeric", length=ncol(preds.oos))
  names(oos.rsrq) <- names(oos.rmse) <- names(oos.acc) <- names(oos.kappa) <- colnames(preds.oos)
  
  for(i in 1:ncol(preds.oos)){
    if(is.categ){
      tbl <- table(preds.oos[,i], as.numeric(ds$pred))
      stats <- confusionMatrix(tbl)
      oos.acc[i] <- stats$overall[1] # accuracy
      oos.kappa[i] <- stats$overall[2] # kappa
    } else {
      smry <- summary(lm(preds.oos[,i] ~ as.numeric(ds$pred)))
      oos.rmse[i] <- sqrt(mean((preds.oos[,i]-as.numeric(ds$pred))^2, na.rm = TRUE))
      oos.rsrq[i] <- smry$r.squared
    }
  }
  
  if(is.categ){
    out <- list(OOSAccuracy=oos.acc,
                MaxCVAccuracy=apply(accur, 2, mean),
                OOSKappa=oos.kappa,
                MaxCVKappa=apply(kappa, 2, mean),
                predictedObs=preds.oos)
  } else{
    out <- list(OOSRMSE=oos.rmse,
                MinCVRMSE=apply(rmse, 2, mean),
                OOSRsquared=oos.rsrq,
                MaxCVRquared=apply(rsrq, 2, mean),
                predictedObs=preds.oos)
  }
  
  return(out)
}


#' @title Nested CV with ensemble modeling.
#'
#' @description
#' \code{caretEnsNestedCV} Prototype ensemble modeling with nested CV>
#' 
#' @return Performance values.
caretEnsNestedCV <- function(ml.out, cv.folds, ds){
  
  if("caretEnsemble" %in% rownames(installed.packages())){
    suppressMessages(library(caretEnsemble))
  } else{
    stop("CaretEnsemble is not installed.")
  }
  
  probs.final <- rep(NA, length=nrow(ds))
  
  # determine whether we are doing classification or regression
  is.categ <- sum(c("Accuracy", "Kappa") %in% colnames(ml.out[[1]][[1]]$results))>0
  
  for(i in 1:length(ml.out)){
    
    tmp.preds <- matrix(NA, nrow=length(unlist(cv.folds[i])), ncol=length(ml.out[[i]]))
    colnames(tmp.preds) <- names(ml.out[[i]])
    
    if(is.categ){
      # for classification
      for(j in 1:length(ml.out[[1]])){
        tmp.preds[,j] <- predict(ml.out[[i]][[j]], newdata=ds[unlist(cv.folds[i]),], type="prob")[,1]     
      }
    } else{
      # for regression
      for(j in 1:length(ml.out[[1]])){
        tmp.preds[,j] <- predict(ml.out[[i]][[j]], newdata=ds[unlist(cv.folds[i]),])    
      }
    }
    
    greedy_ensemble <- caretEnsemble(ml.out[[i]])
    
    wts <- greedy_ensemble$weights
    
    for(k in 1:length(ml.out[[i]])){
      if(colnames(tmp.preds)[k] %in% names(wts)){
        tmp.preds[,k] <- tmp.preds[,k]*wts[colnames(tmp.preds)[k]]
      } else{
        tmp.preds[,k] <- tmp.preds[,k]*0
      }
    }
    
    probs.final[unlist(cv.folds[i])] <- apply(tmp.preds, 1, sum)
  }
  
  if(is.categ){
    # for classification
    tbl <- table(ifelse(probs.final<0.5, 2, 1), as.numeric(ds$pred))
    stats <- confusionMatrix(tbl)
    
    out <- list(OOSAccuracy=stats$overall[1],
                OOSKappa=stats$overall[2])
  } else{
    # for regression analysis
    out <- list(OOSRsquared=summary(lm(probs.final ~ as.numeric(ds$pred)))$r.squared,
                OOSRMSE=sqrt(mean((probs.final-as.numeric(ds$pred))^2, na.rm = TRUE)))
  }
  
  return(out)
}

#' @title Convert covariates for machine learning.
#'
#' @description
#' \code{covarConvert} takes none numeric and categories and converts them all for learning.
#' 
#' @param clin.covars input data.frame 
#' @return model matrix for learning.
covarConvert <- function(clin.covars){
  
  # this function takes a factor and numeric matrix and imputes the numeric data and converts the factors to model matrix and combines everything together for ML
  
  if("caret" %in% rownames(installed.packages())){
    suppressMessages(library("caret"))
  } else {
    print("Caret package was detected as installed. Would you like to install it? (y/n)")
    usr.resp <- readline()
    if(tolower(usr.resp)=="y"){
      install.packages("caret")
    } else {
      stop("Install package before proceeding.")
    }
  }
  
  # make model matrix
  options(na.action="na.pass")
  
  # get the class of each seperate column of the data frame
  classes <- sapply(1:ncol(clin.covars), function(x) class(clin.covars[,x]))
  
  # seperate them into two different groups because the numeric classes have NAs and preProcess can only take numeric, not factors or characaters
  fact.clin.data <- clin.covars[,classes=="factor" | classes=="character"]
  numeric.clin.data <- clin.covars[,classes=="integer" | classes=="numeric"]
  
  #  confirm that the seperated data is the same size as the original data
  stopifnot(ncol(clin.covars) == ncol(numeric.clin.data) + ncol(fact.clin.data))
  
  impute.data <- preProcess(numeric.clin.data,
                            k=5,                   # number of nearest neighbors to use for KNN impute
                            method="medianImpute") # other valid options include "knnImpute" and "bagImpute"
  
  # apply the imputation to get the data frame back
  numeric.imput.data <- predict(impute.data, numeric.clin.data)
  
  # cbind the data together and make a model matrix out of it - drop column labeling SLE vs healthy because factors have to have multiple levels
  all.clin <- model.matrix(~., data=cbind(fact.clin.data, numeric.imput.data))
  
  return(all.clin)
  
}

#' @title Plot VIP values.
#'
#' @description
#' \code{plotVIP} Make a plot for the top VIP.
#' 
#' @param model to use
#' @param nFeats.plot number of features to plot
#' @return a barplot
plotVIP <- function(model, nFeats.plot){
  var.imp.ord <- order(model$importance$Overall, decreasing=TRUE)
  varImp.plotData <- model$importance$Overall[var.imp.ord]
  names(varImp.plotData)  <- rownames(model$importance)[var.imp.ord]
  
  par(mar=c(12,4,2,2))
  barplot(varImp.plotData[1:nFeats.plot], las=2)
}

#' @title Extract out of sample CV predictions.
#'
#' @description
#' \code{extractCVres} This extracts the out of sample cross validated predictions from the caret model object.
#' 
#' @param mod a caret model
#' @param average TRUE or FALSE; averages over repeats if TRUE
#' @return data frame of predictions
extractCVres <- function(mod, average=TRUE){
  # this function extracts the probabilties and predicted values for the cross validated results  
  # this matrix in the number of samples x repeats long
  
  # initialize empty matrix
  logic.mat <- matrix(NA, nrow=nrow(mod$pred), ncol=length(mod$bestTune))
  # loop over every tuning parameter creating matrix of logical values
  for(i in 1:length(mod$bestTune)){
    tmp <- mod$pred[,names(mod$bestTune[i])] == unlist(mod$bestTune[i])
    tmp[is.na(tmp)] <- FALSE
    logic.mat[,i] <- tmp
  }
  
  if(average){
    ret <- processCVres(mod$pred[apply(logic.mat, 1, all),])
  } else{
    ret <- mod$pred[apply(logic.mat, 1, all),]
  }
  
  return(ret)
}



#' Dynamic logistic state-space prediction model for binary outcomes
#' @author Jiakun Jiang, Wei Yang, Stephen E. Kimmel and Wensheng Guo
#' @description Implements dynamic logistic state-space prediction model for binary outcomes as described in Jiakun et al.(2021, Biometrics). In retrospective study,
#' it is suitable to use the main function DLSSM with specify trainning sample size. The results can also be repeated using subfunctions which is more suitable for online implementation. The algorithm was composed by training part and validation part.
#' On the stage of training, the smoothing parameters were selected by maximizing likelihood function. Then, based on the estimated smoothing parameters, run the Kalman filtering and
#' smoothing algorithm to estimate both the time-varying and time-invariant coefficients in the model. Based on the estimated coefficients and state-sapce model, it was straightforward to do prediction.
#' @export
#' @import Matrix
#' @param x n by q matrix of covariates
#' @param y A vector of binary outcome of length n
#' @param t A vector of observational timepoints of length n
#' @param S Number of batches (equally spaced)
#' @param training_samp Number of batches used as training data. If null (default), half number of batches will be used.
#' @param vary A vector specify the covariates with time-varying coefficients. The remaining covariates have constant coefficients. For example, vector vary=(1,2) will specify first and third covariates as having time-varying coefficients. Intercept is always time-varying. If vary=NULL, only intercept is time-varying coefficient.
#' @param autotune T/F indicates whether or not the automatic tuning procedure described in Jiakun et al. (2021) should be applied.  Default is true.
#' @param Lambda Specify smoothing parameters manually if autotune=F
#' @param K Specify how many steps ahead prediction of coefficients and probabilities
#' @details User first need to identify the covariates which have time-varying coefficients.
#' User need to decide the number of batches S which is achieved by dividing the observational time domain time into equally spaced time intervals. The number of batches S should satisfy a condition that all intervals have data.
#' The selection smoothing parameters usually be recommended by maximizing likelihood. The training data should have relatively large sample size to ensure the tuned smoothing parameters reliable.
#' @return A list with letters and numbers.
#'  \tabular{ll}{
#'    \code{Pred} \tab Predicted coefficients in the prediction step of Kalman Filter \cr
#'    \tab \cr
#'    \code{Pred.var} \tab Covariance matrix of Predicted coefficient in prediction step of Kalman Filter \cr
#'    \tab \cr
#'    \code{Filter} \tab Filtered coefficients in Kalman Filter \cr
#'    \tab \cr
#'    \code{Filter.var} \tab Covariance matrix of filtered coefficients in Kalman Filter \cr
#'    \tab \cr
#'    \code{Smooth} \tab Smoothing of coefficients \cr
#'    \tab \cr
#'    \code{Smooth.var} \tab Covariance matrix of smoothing of coefficients \cr
#'    \tab \cr
#'    \code{Pred.K} \tab K-steps ahead prediction of coefficients \cr
#'    \tab \cr
#'    \code{Pred.K.var} \tab Covariance matrix of K-steps ahead of prediction of coefficients \cr
#'    \tab \cr
#'    \code{Lambda} \tab Smoothing parameters \cr
#'    \tab \cr
#'    \code{q} \tab Number of covariates \cr
#'    \tab \cr
#'    \code{q1} \tab Number of covariates with varying coefficients \cr
#'    \tab \cr
#'    \code{train.time} \tab Time-points of training data \cr
#'    \tab \cr
#'    \code{dim} \tab Number of varying coefficients which equals to q+1(including a varying intercept) \cr
#'    \tab \cr
#'    \code{dim.con} \tab Number of constant coefficients \cr
#'    \tab \cr
#'    \code{TT} \tab Transformation matrix \cr
#'    \tab \cr
#'    \code{Q} \tab Variance matrix \cr
#'    \tab \cr
#'    \code{Prob.pred.K} \tab K-steps ahead prediction of probabilities \cr
#'    \tab \cr
#'    ...
#'  }
#' @examples
#' rm(list=ls())
#' set.seed(12345)
#' n=8000
#' beta0=function(t)   0.1*t-1   # Intercept
#' beta1=function(t)  cos(2*pi*t)   # Varying coefficient
#' beta2=function(t)  sin(2*pi*t)   # Varying coefficient
#' alph1=alph2=1
#' x=matrix(runif(n*4,min=-4,max=4),nrow=n,ncol=4)
#' t=sort(runif(n))
#' coef=cbind(beta0(t),beta1(t),beta2(t),rep(alph1,n),rep(alph2,n))
#' covar=cbind(rep(1,n),x)
#' linear=apply(coef*covar,1,sum)
#' prob=exp(linear)/(1+exp(linear))
#' y=as.numeric(runif(n)<prob)
#' fit=DLSSM(x,y,t,S=100,vary=c(1,2),autotune=TRUE,training_samp=75,K=1)
#'
#' # plot one-step ahead predicted, filtered and smoothed cofficients
#' # DLSSM.plot(fit)
#' # fit$Lambda
#' # hist(fit$Prob.pred.K[[75]][[1]],main="Histogram of predicted
#' # probabilities of subjects in 76-th batch", xlab = "Probability")
#'
#'
#' # Implement the DLSSM in a "streaming" model
#' S=100
#' data.batched=Data.batched(x,y,t,S)
#' # Using DLSSM.init() on training dataset (first S.initial batches of data) to tune smoothing parameters
#' init.fit=DLSSM.init(data.batched,S.initial=75,vary=c(1,2),autotune=TRUE)
#' Prediction=matrix(NA,S,2*3+2)
#' Prediction.var=array(NA,dim=c(S,2*3+2,2*3+2))
#' Filtering=matrix(NA,S,2*3+2)
#' Filtering.var=array(NA,dim=c(S,2*3+2,2*3+2))
#' Prediction[1:75,]=init.fit$Pred
#' Prediction.var[1:75,,]=init.fit$Pred.var
#' Filtering[1:75,]=init.fit$Filter
#' Filtering.var[1:75,,]=init.fit$Filter.var
#'
#' # The following streaming structure fit online dynamic implementation
#' for(i in 76:100){
#'   pred1=DLSSM.predict(init.fit,newx=NULL,K=1)
#'   Prediction[i,]=pred1$coef.pred
#'   Prediction.var[i,,]=pred1$coef.pred.var
#'   init.fit=DLSSM.filter(init.fit,data.batched$x.batch[[i]],data.batched$y.batch[[i]])
#'   Filtering[i,]=init.fit$Filter
#'   Filtering.var[i,,]=init.fit$Filter.var
#' }
#' # Smoothed results be generated by integrating historical prediction and filtering results
#' Smoothed=DLSSM.smooth(init.fit,Prediction,Prediction.var,Filtering,Filtering.var)
DLSSM<-function(x,y,t,S,vary,autotune=TRUE,training_samp,Lambda=NULL,K){
  ###check data
  n=dim(x)[1]              #sample size  #num.vary
  q=dim(x)[2]              #dimension of covariates
  num.vary=q1=length(vary) #number of covariates with varying coefficient
  q2=q-q1                  #number of covariates with time-invariant coefficient
  if(n!=length(t)|n!=length(y)){
    stop("The length of data does not match.")
  }
  if(any(is.na(x))==TRUE|any(is.na(t))==TRUE|any(is.na(y))==TRUE){
    stop("There has NA in the data")
  }
  if(is.vector(vary)==FALSE & is.null(vary)==FALSE){
    stop("vary need to be a positive integer vector or NULL")
  }
  if(is.null(vary)==FALSE){
    if(max(vary)>q|min(vary)<0|min(vary)==0) {
      stop("If vary is not NULL. The elements in vary need to be between 1 and number of covariates.")
    }
  }
  if(is.numeric(x)==FALSE|is.numeric(y)==FALSE|is.numeric(t)==FALSE){
    stop("x,y,t need to be numeric")
  }
  if(num.vary>q){
    stop("The number of time-varying coefficients need to smaller than number of covariates.")
  }
  if(as.integer(abs(S))!=S){
    stop("S need to be a positive integer")
  }
  if(training_samp>S){
    stop("The number of batches used as training data should not excess total batches S.")
  }

  t=t/max(t)
  oder=order(t)
  t.order=t[oder]
  x.order=x[oder,]
  y.order=y[oder]
  # create batch
  if(num.vary==q){
    x.vary=x.order[,1:q]
    t=list()         # observational time-points
    X=list()         # Covariates corresponding to varying coefficients
    #XX=list()        # Covariates corresponding to constant coefficients
    y=list()         # binary outcome
    Data.check=rep(NA,S)
    for(i in 1:S){
      sel=t.order>(i-1)/S & t.order<=i/S
      t[[i]]=t.order[sel]
      X[[i]]=x.vary[sel,]
      y[[i]]=as.numeric(y.order[sel])
      Data.check[i]=sum(sel)
    }
  }
  if(num.vary>0&num.vary<q){
    x.vary=as.matrix(x.order[,vary],ncol=num.vary)
    x.constant=as.matrix(x.order[,setdiff(1:q,vary)],ncol=q-num.vary)
    t=list()         # observational time-points
    X=list()         # Covariates corresponding to varying coefficients
    XX=list()        # Covariates corresponding to constant coefficients
    y=list()         # binary outcome
    Data.check=rep(NA,S)
    for(i in 1:S){
      sel=t.order>(i-1)/S & t.order<=i/S
      t[[i]]=t.order[sel]
      X[[i]]=x.vary[sel,]
      XX[[i]]=x.constant[sel,]
      y[[i]]=as.numeric(y.order[sel])
      Data.check[i]=sum(sel)
    }
  }
  if(num.vary==0){  # no varying-coefficient
    x.constant=x.order
    t=list()         # observational time-points
    XX=list()        # Covariates corresponding to constant coefficients
    y=list()         # binary outcome
    Data.check=rep(NA,S)
    for(i in 1:S){
      sel=t.order>(i-1)/S & t.order<=i/S
      t[[i]]=t.order[sel]
      XX[[i]]=x.constant[sel,]
      y[[i]]=as.numeric(y.order[sel])
      Data.check[i]=sum(sel)
    }
  }
  if(any(Data.check==0)){
    stop("There exist some intervals [(i-1)/S,i/S] with no data, please reduce numder of batches")
  }
  #########################################
  #State space model setup
  #########################################
  dim=q1+1                                  # Number of varying coefficients including intercept
  dim.con=q2                                # Number of constant coefficients
  TT=matrix(0,2*dim+dim.con,2*dim+dim.con)  # Transformation matrix in state-space model
  QQ=matrix(NA,2,2)                         # Covariance matrix in state-space model
  delta.t=1/S                               # Batch data be generated equally spaced
  TTq1=matrix(0,2*dim,2*dim)                # Transformation matrix for varying coefficient
  TTq2=diag(rep(1,dim.con))                 # Transformation matrix for constant coefficient
  for(ss in 1:dim){
    TTq1[(2*ss-1):(2*ss),(2*ss-1):(2*ss)]=matrix(c(1, 0, delta.t, 1),2, 2)
  }
  TT=as.matrix(bdiag(TTq1,TTq2))
  QQ=matrix(c((delta.t)^3/3,(delta.t)^2/2,(delta.t)^2/2,delta.t),2,2)
  a1=rep(0,2*dim+dim.con)        # Initial value
  diff=log(100)                  # Diffuse variance of initial prior distribution
  n.train=training_samp            # number of batches
  #########################################
  #Likelihood function to tune smoothing parameters Lambda based on training data
  #########################################
  if(autotune==TRUE){
    likehood.predictive=function(xx) {
      Lambda=xx
      Q=matrix(0,2*dim+dim.con,2*dim+dim.con)
      Q1=matrix(0,2*dim,2*dim)
      for(ss in 1:dim){
        Q1[(2*ss-1):(2*ss),(2*ss-1):(2*ss)]=exp(Lambda[ss])*QQ
      }
      Q2=TTq2=matrix(0,dim.con,dim.con)
      Q=as.matrix(bdiag(Q1,Q2))

      P1 <- exp(diff)*diag(2*dim+dim.con)

      Sample.likehood=rep(NA,n.train)

      prediction=matrix(NA,n.train,dim*2+dim.con)
      prediction.var=array(NA,dim=c(n.train,dim*2+dim.con,dim*2+dim.con))
      prediction[1,]=a1
      prediction.var[1,,]=P1


      filter=matrix(NA,n.train,dim*2+dim.con)
      filter.var=array(NA,dim=c(n.train,dim*2+dim.con,dim*2+dim.con))

      intial=prediction[1,]
      ZZ=matrix(0,2*dim+dim.con,length(t[[1]]))
      if(num.vary==q){
        ZZ[1,]=1
        for(sss in 1:(dim-1)) {
          ZZ[2*sss+1,]=t(X[[1]])[sss,]
        }
      }
      if(num.vary>0&num.vary<q){
        ZZ[1,]=1
        for(sss in 1:(dim-1)) {
          ZZ[2*sss+1,]=t(X[[1]])[sss,]
        }
        ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[1]])
      }
      if(num.vary==0){
        ZZ[1,]=1
        ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[1]])
      }

      for(inter in 1:1000){
        summa=rep(0,2*dim+dim.con)
        summa.cov=matrix(0,2*dim+dim.con,2*dim+dim.con)
        for(j in 1:length(t[[1]])){
          y.hat=exp(ZZ[,j]%*%intial)/(1+exp(ZZ[,j]%*%intial))
          summa=summa+as.numeric(y[[1]][j]-y.hat)*ZZ[,j]
          summa.cov=summa.cov+as.numeric(y.hat*(1-y.hat))*ZZ[,j]%o%ZZ[,j]
        }

        SCORE=summa-t(solve(prediction.var[1,,])%*%(intial-matrix(prediction[1,],2*dim+dim.con,1)))
        COVMATRIX=-solve(prediction.var[1,,])-summa.cov
        recur=intial-(0.2*solve(COVMATRIX)%*%t(SCORE))
        befor=intial
        intial=recur
        if(mean(abs(befor-recur))<0.00001){break}
      }
      filter[1,]=recur
      filter.var[1,,]=solve(-COVMATRIX)


      lihood=0
      for(j in 1:length(t[[1]])){
        y.hatt=exp(ZZ[,j]%*%filter[1,])/(1+exp(ZZ[,j]%*%filter[1,]))
        Covmatrix=-solve(prediction.var[1,,])-as.numeric(y.hatt*(1-y.hatt))*ZZ[,j]%o%ZZ[,j]
        densi=1/sqrt(det(prediction.var[1,,]))*
          exp(-0.5*(filter[1,]-prediction[1,])%*%solve(prediction.var[1,,])%*%(filter[1,]-prediction[1,]))
        lihood=lihood+log(2*pi*sqrt(det(solve(-Covmatrix)))*(y.hatt^(y[[1]][j]))*((1-y.hatt)^(1-y[[1]][j]))*densi)
      }
      Sample.likehood[1]=lihood

      for(i in 2:n.train) {
        prediction[i,]=TT%*%filter[i-1,]
        prediction.var[i,,]=TT%*%filter.var[i-1,,]%*%t(TT)+Q

        intial=prediction[i,]
        ZZ=matrix(0,dim*2+dim.con,length(t[[i]]))
        if(num.vary==q){
          ZZ[1,]=1
          for(sss in 1:(dim-1)) {
            ZZ[2*sss+1,]=t(X[[i]])[sss,]
          }
        }
        if(num.vary>0&num.vary<q){
          ZZ[1,]=1
          for(sss in 1:(dim-1)) {
            ZZ[2*sss+1,]=t(X[[i]])[sss,]
          }
          ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[i]])
        }
        if(num.vary==0){
          ZZ[1,]=1
          ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[i]])
        }

        for(inter in 1:1000){
          summa=rep(0,dim*2+dim.con)
          summa.cov=matrix(0,dim*2+dim.con,dim*2+dim.con)
          for(j in 1:length(t[[i]])){
            y.hat=exp(ZZ[,j]%*%intial)/(1+exp(ZZ[,j]%*%intial))
            summa=summa+as.numeric(y[[i]][j]-y.hat)*ZZ[,j]
            summa.cov=summa.cov+as.numeric(y.hat*(1-y.hat))*ZZ[,j]%o%ZZ[,j]
          }
          SCORE=summa-t(solve(prediction.var[i,,])%*%(intial-matrix(prediction[i,],2*dim+dim.con,1)))
          COVMATRIX=-solve(prediction.var[i,,])-summa.cov
          recur=intial-(0.2*solve(COVMATRIX)%*%t(SCORE))
          befor=intial
          intial=recur
          if(max(abs(befor-recur))<0.00001){break}
        }
        filter[i,]=recur
        filter.var[i,,]=solve(-COVMATRIX)

        lihood=0
        for(j in 1:length(t[[i]])){
          y.hatt=exp(ZZ[,j]%*%filter[i,])/(1+exp(ZZ[,j]%*%filter[i,]))
          Covmatrix=-solve(prediction.var[i,,])-as.numeric(y.hatt*(1-y.hatt))*ZZ[,j]%o%ZZ[,j]
          densi=1/sqrt(det(prediction.var[i,,]))*
            exp(-0.5*(filter[i,]-prediction[i,])%*%solve(prediction.var[i,,])%*%(filter[i,]-prediction[i,]))
          lihood=lihood+log(2*pi*sqrt(det(solve(-Covmatrix)))*(y.hatt^(y[[i]][j]))*((1-y.hatt)^(1-y[[i]][j]))*densi)
        }
        Sample.likehood[i]=lihood
      }
      likelihood=-sum((Sample.likehood))
      return(likelihood)
    }
    if(dim>1){
      search.opt=optim(rep(2,dim),likehood.predictive)  #Optimize likelihood function
      Lambda=exp(search.opt$par)
    }
    if(dim==1){
      search.opt=optimize(likehood.predictive,c(-20,20))  #Optimize likelihood function
      Lambda=exp(search.opt$minimum)
    }
    #Smoothing parameter for intercept
  }else{
    if(length(Lambda)!=num.vary+1){
      stop("The number of smoothing parameters should equal to number of varying coefficients, including varying intercept.")
    }
    Lambda=Lambda
  }

  P1<-exp(diff)*diag(dim*2+dim.con)                   #Diffuse prior
  Q=matrix(0,dim*2+dim.con,dim*2+dim.con)

  Q1=matrix(0,2*dim,2*dim)
  for(ss in 1:dim){
    Q1[(2*ss-1):(2*ss),(2*ss-1):(2*ss)]=Lambda[ss]*QQ
  }
  Q2=TTq2=matrix(0,dim.con,dim.con)
  Q=as.matrix(bdiag(Q1,Q2))

  #########################################
  #Model estimation using all data with Kalman filter
  #########################################
  prediction.K=list()  #for K steps ahead prediction
  prediction.var.K=list()  #for K steps ahead prediction
  prediction=matrix(NA,S,2*dim+dim.con)                    # Mean prediction
  prediction.var=array(NA,dim=c(S,2*dim+dim.con,2*dim+dim.con))  # Covariance prediction
  prediction[1,]=a1                                  # Initial mean
  prediction.var[1,,]=P1                             # Initial covariance

  ##K-steps ahead prediction
  prediction.K[[1]]=a1
  prediction.var.K[[1]]=P1

  filter=matrix(NA,S,2*dim+dim.con)                        # Filtering mean
  filter.var=array(NA,dim=c(S,2*dim+dim.con,2*dim+dim.con))      # Filtering covariance

  #Kalman filter on first timepoint
  intial=prediction[1,]
  ZZ=matrix(0,2*dim+dim.con,length(t[[1]]))  #define covariate z in our state space model (5)
  if(num.vary==q){
    ZZ[1,]=1
    for(sss in 1:(dim-1)) {
      ZZ[2*sss+1,]=t(X[[1]])[sss,]
    }
  }
  if(num.vary>0&num.vary<q){
    ZZ[1,]=1
    for(sss in 1:(dim-1)) {
      ZZ[2*sss+1,]=t(X[[1]])[sss,]
    }
    ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[1]])
  }
  if(num.vary==0){
    ZZ[1,]=1
    ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[1]])
  }
  for(inter in 1:1000){       # Newton-Raphson iterative algorithm to get filtering estimator
    summa=rep(0,2*dim+dim.con)
    summa.cov=matrix(0,2*dim+dim.con,2*dim+dim.con)
    for(j in 1:length(t[[1]])){
      y.hat=exp(ZZ[,j]%*%intial)/(1+exp(ZZ[,j]%*%intial))
      summa=summa+as.numeric(y[[1]][j]-y.hat)*ZZ[,j]
      summa.cov=summa.cov+as.numeric(y.hat*(1-y.hat))*ZZ[,j]%o%ZZ[,j]
    }
    SCORE=summa-t(solve(prediction.var[1,,])%*%(intial-matrix(prediction[1,],2*dim+dim.con,1)))
    COVMATRIX=-solve(prediction.var[1,,])-summa.cov
    recur=intial-(0.2*solve(COVMATRIX)%*%t(SCORE))
    befor=intial
    intial=recur
    if(mean(abs(befor-recur))<0.00001){break}
    if(inter==1000){stop("failed converge","\n")}
  }
  filter[1,]=recur                      # Filtering mean
  filter.var[1,,]=solve(-COVMATRIX)     # Filtering covariance

  #Kalman filter on timepoints from 2 to S
  Prob.est=list()   # one-step ahead prediction
  Prob.est.K=list() # K steps ahead prediction
  for(i in 2:S) {
    if(S-i>1){
      K.S.P=matrix(NA,min(S-i,K),2*dim+dim.con)
      K.S.P.var=array(NA,dim=c(min(S-i,K),2*dim+dim.con,2*dim+dim.con))
      K.S.P[1,]=TT%*%filter[i-1,]
      K.S.P.var[1,,]=TT%*%filter.var[i-1,,]%*%t(TT)+Q
      if(K>1){
        for(kp in 2:min(S-i,K)){
          K.S.P[kp,]=TT%*%K.S.P[kp-1,]
          K.S.P.var[kp,,]=TT%*%K.S.P.var[kp-1,,]%*%t(TT)+Q
        }
      }
      prediction.K[[i]]=K.S.P
      prediction.var.K[[i]]=K.S.P.var
    }
    if(S-i==1){
      prediction.K[[i]]=matrix(TT%*%filter[i-1,],nrow=1)
      AAAA=array(NA,dim=c(1,2*dim+dim.con,2*dim+dim.con))
      AAAA[1,,]=TT%*%filter.var[i-1,,]%*%t(TT)+Q
      prediction.var.K[[i]]=AAAA
    }

    #K-steps ahead prediction of probability.
    if(S-i>0){
      y.hat.k=list()
      for(kspp in 1:dim(prediction.K[[i]])[1]){
        intial=prediction.K[[i]][kspp,]

        ZZ=matrix(0,dim*2+dim.con,length(t[[i+kspp-1]]))
        if(num.vary==q){
          ZZ[1,]=1
          for(sss in 1:(dim-1)) {
            ZZ[2*sss+1,]=t(X[[i+kspp-1]])[sss,]
          }
        }
        if(num.vary>0&num.vary<q){
          ZZ[1,]=1
          for(sss in 1:(dim-1)) {
            ZZ[2*sss+1,]=t(X[[i+kspp-1]])[sss,]
          }
          ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[i+kspp-1]])
        }
        if(num.vary==0){
          ZZ[1,]=1
          ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[i+kspp-1]])
        }
        y.hat.k[[kspp]]=exp(t(ZZ)%*%intial)/(1+exp(t(ZZ)%*%intial))
      }
      Prob.est.K[[i]]=y.hat.k
    }
    # Prediction step
    prediction[i,]=TT%*%filter[i-1,]
    prediction.var[i,,]=TT%*%filter.var[i-1,,]%*%t(TT)+Q
    # Filtering step
    intial=prediction[i,]
    ZZ=matrix(0,dim*2+dim.con,length(t[[i]]))
    if(num.vary==q){
      ZZ[1,]=1
      for(sss in 1:(dim-1)) {
        ZZ[2*sss+1,]=t(X[[i]])[sss,]
      }
    }
    if(num.vary>0&num.vary<q){
      ZZ[1,]=1
      for(sss in 1:(dim-1)) {
        ZZ[2*sss+1,]=t(X[[i]])[sss,]
      }
      ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[i]])
    }
    if(num.vary==0){
      ZZ[1,]=1
      ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[i]])
    }

    if(i>n.train){
      y.hat=exp(t(ZZ)%*%intial)/(1+exp(t(ZZ)%*%intial))
      Prob.est[[i-n.train]]=y.hat
    }




    for(inter in 1:1000){     # Newton-Raphson
      summa=rep(0,dim*2+dim.con)
      summa.cov=matrix(0,dim*2+dim.con,dim*2+dim.con)
      for(j in 1:length(t[[i]])){
        y.hat=exp(ZZ[,j]%*%intial)/(1+exp(ZZ[,j]%*%intial))
        summa=summa+as.numeric(y[[i]][j]-y.hat)*ZZ[,j]
        summa.cov=summa.cov+as.numeric(y.hat*(1-y.hat))*ZZ[,j]%o%ZZ[,j]
      }
      SCORE=summa-t(solve(prediction.var[i,,])%*%(intial-matrix(prediction[i,],2*dim+dim.con,1)))
      COVMATRIX=-solve(prediction.var[i,,])-summa.cov
      recur=intial-(0.2*solve(COVMATRIX)%*%t(SCORE))
      befor=intial
      intial=recur
      if(mean(abs(befor-recur))<0.00001){break}
      if(inter==1000){stop("Newton-Raphson failed to converge","\n")}
    }
    filter[i,]=recur
    filter.var[i,,]=solve(-COVMATRIX)
  }

  #Smoothed coefficients on training data
  state.smooth=matrix(NA,S,2*dim+dim.con)
  F.t.smooth=array(NA,dim=c(S,2*dim+dim.con,2*dim+dim.con))
  state.smooth[S,]=filter[S,]
  F.t.smooth[S,,]=filter.var[S,,]
  for(i in (S-1):1){
    A.i=filter.var[i,,]%*%t(TT)%*%solve(prediction.var[i+1,,])
    state.smooth[i,]=filter[i,]+A.i%*%(state.smooth[i+1,]-prediction[i+1,])
    F.t.smooth[i,,]=filter.var[i,,]-A.i%*%(prediction.var[i+1,,]-F.t.smooth[i+1,,])%*%t(A.i)
  }
  Est=list(Pred=prediction,Pred.var=prediction.var,Filter=filter
           ,Filter.var=filter.var,Smooth=state.smooth,Smooth.var=F.t.smooth
           ,Lambda=Lambda,vary=vary,q=q,q1=q1,q2=q2
           ,train.time=t.order,dim=dim,dim.con=dim.con,TT=TT,Q=Q,Pred.K=prediction.K,Pred.K.var=prediction.var.K
           ,Prob.pred=Prob.est.K,Prob.pred.K=Prob.est.K,S=S,gap.len=delta.t,training_samp=training_samp)
  return(Est)
}

#' Combine data into Batched data
#' @author Jiakun Jiang, Wei Yang, Stephen E. Kimmel and Wensheng Guo
#' @description The time domain of oberservation was first standardized in [0,1]. Then [0,1] was divided into S equally spaced intervals as described in Jiakun et al.(2021, Biometrics).
#' Then data fall into each interval compose a batch of data.
#' @param x n by q matrix of covariates
#' @param y n vector of binary responses
#' @param t n vector of observed time-points
#' @param S number of batches
#' @export
#' @return A list of batched data.
#'  \tabular{ll}{
#'    \code{x.batch} \tab Batched covariates \cr
#'    \tab \cr
#'    \code{y.batch} \tab Batched binary outcome \cr
#'    \tab \cr
#'    \code{t.batch} \tab Batched time-points \cr
#'    \tab \cr
#'    \code{gap.len} \tab Length of the equally spaced time interval \cr
#'  }
Data.batched<-function(x,y,t,S){
  ###check data
  n=dim(x)[1]              #sample size
  q=dim(x)[2]              #dimension of covariates
  if(n!=length(t)|n!=length(y)){
    stop("The dimension of data does not match.")
  }
  if(any(is.na(x))==TRUE|any(is.na(t))==TRUE|any(is.na(y))==TRUE){
    stop("There has NA in the data")
  }
  if(as.integer(abs(S))!=S){
    stop("S need to be a positive integer")
  }
  t=t/max(t)
  oder=order(t)
  t.order=t[oder]
  x.order=x[oder,]
  y.order=y[oder]
  # create batch
  t.batch=list()         #Observational time-points
  x.batch=list()         #Covariates corresponding to varying coefficients
  y.batch=list()         #Binary outcome
  Data.check=rep(NA,S)
  for(i in 1:S){
    sel=t.order>(i-1)/S & t.order<=i/S
    t.batch[[i]]=t.order[sel]
    x.batch[[i]]=x.order[sel,]
    y.batch[[i]]=as.numeric(y.order[sel])
    Data.check[i]=sum(sel)
  }
  if(any(Data.check==0)){
    stop("There exist some intervals [(i-1)/S,i/S] with no data, please adjust batches")
  }
  return(list(x.batch=x.batch,y.batch=y.batch,t.batch=t.batch,gap.len=1/S))
}

#' Smoothing parameters selection and inital model fitting
#' @author Jiakun Jiang, Wei Yang, Stephen E. Kimmel and Wensheng Guo
#' @description This function is for tuning smoothing parameters using training data. The likelihood was calculated by Kalman Filter and maximized to estimate the smoothing parameters.
#' For the given smoothing parameters, the model coefficients can
#' be efficiently estimated using a Kalman filtering algorithm.
#' @param data.batched A object generated by function Data.batched()
#' @param S.initial How many batches of data to be used to tuning smoothing parameters
#' @param vary A vector specify the covariates with time-varying coefficients. The remaining covariates have constant coefficients. For example, vectorvary=(1,2) will specify first two covariates as having time-varying coefficients. Intercept is always time-varying. If vary=NULL, only intercept is time-varying coefficient.
#' @param autotune T/F indicates whether or not the automatic tuning procedure desribed in Jiakun et al. (2021) should be applied.  Default is true.
#' @param Lambda specify smoothing parameters if autotune=F
#' @export
#' @return See the returned value of DLSSM().
DLSSM.init<-function(data.batched,S.initial,vary,autotune=TRUE,Lambda=NULL){
  x.b=data.batched$x.batch
  y.b=data.batched$y.batch
  t.b=data.batched$t.batch
  q=dim(x.b[[1]])[2]      #dimension of covariates filter
  q1=num.vary=length(vary)    #number of covariates with varying coefficient
  q2=q-q1                  #number of covariates with time-invariant coefficient
  S0=S.initial

  if(num.vary>0){
    t=list()         # observational time-points
    X=list()         # Covariates corresponding to varying coefficients
    XX=list()        # Covariates corresponding to constant coefficients
    y=list()         # binary outcome
    for(i in 1:S0){
      t[[i]]=t.b[[i]]
      X[[i]]=x.b[[i]][,vary]
      XX[[i]]=x.b[[i]][,setdiff(1:q,vary)]
      y[[i]]=as.numeric(y.b[[i]])
    }
  }
  if(num.vary==0){   # no varying-coefficient
    t=list()
    XX=list()
    y=list()
    for(i in 1:S0){
      t[[i]]=t.b[[i]]
      XX[[i]]=x.b[[i]]
      y[[i]]=as.numeric(y.b[[i]])
    }
  }
  dim=q1+1                                  # Number of varying coefficients including intercept
  dim.con=q2                                # Number of constant coefficients
  TT=matrix(0,2*dim+dim.con,2*dim+dim.con)  # Transformation matrix in state-space model
  QQ=matrix(NA,2,2)                         # Covariance matrix in state-space model
  delta.t=data.batched$gap.len                               # Batch data be generated equally spaced
  TTq1=matrix(0,2*dim,2*dim)                # Transformation matrix for varying coefficient
  TTq2=diag(rep(1,dim.con))                 # Transformation matrix for constant coefficient
  for(ss in 1:dim){
    TTq1[(2*ss-1):(2*ss),(2*ss-1):(2*ss)]=matrix(c(1, 0, delta.t, 1),2, 2)
  }
  TT=as.matrix(bdiag(TTq1,TTq2))
  QQ=matrix(c((delta.t)^3/3,(delta.t)^2/2,(delta.t)^2/2,delta.t),2,2)
  a1=rep(0,2*dim+dim.con)        # Initial value
  diff=log(100)                  # Diffuse variance of initial prior distribution
  n.train=S0            # number of batches S
  #########################################
  #Likelihood function to tune smoothing parameters Lambda based on training data
  #########################################
  if(autotune==TRUE){
    likehood.predictive=function(xx) {
      Lambda=xx
      Q=matrix(0,2*dim+dim.con,2*dim+dim.con)
      Q1=matrix(0,2*dim,2*dim)
      for(ss in 1:dim){
        Q1[(2*ss-1):(2*ss),(2*ss-1):(2*ss)]=exp(Lambda[ss])*QQ
      }
      Q2=TTq2=diag(rep(0,dim.con))
      Q=as.matrix(bdiag(Q1,Q2))

      P1 <- exp(diff)*diag(2*dim+dim.con)

      Sample.likehood=rep(NA,n.train)

      prediction=matrix(NA,n.train,dim*2+dim.con)
      prediction.var=array(NA,dim=c(n.train,dim*2+dim.con,dim*2+dim.con))
      prediction[1,]=a1
      prediction.var[1,,]=P1


      filter=matrix(NA,n.train,dim*2+dim.con)
      filter.var=array(NA,dim=c(n.train,dim*2+dim.con,dim*2+dim.con))

      intial=prediction[1,]
      ZZ=matrix(0,2*dim+dim.con,length(t[[1]]))

      if(dim>1){
        ZZ[1,]=1
        for(sss in 1:(dim-1)) {
          ZZ[2*sss+1,]=t(X[[1]])[sss,]
        }
        ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[1]])
      }
      if(dim==1){
        ZZ[1,]=1
        ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[1]])
      }

      for(inter in 1:1000){
        summa=rep(0,2*dim+dim.con)
        summa.cov=matrix(0,2*dim+dim.con,2*dim+dim.con)
        for(j in 1:length(t[[1]])){
          y.hat=exp(ZZ[,j]%*%intial)/(1+exp(ZZ[,j]%*%intial))
          summa=summa+as.numeric(y[[1]][j]-y.hat)*ZZ[,j]
          summa.cov=summa.cov+as.numeric(y.hat*(1-y.hat))*ZZ[,j]%o%ZZ[,j]
        }

        SCORE=summa-t(solve(prediction.var[1,,])%*%(intial-matrix(prediction[1,],2*dim+dim.con,1)))
        COVMATRIX=-solve(prediction.var[1,,])-summa.cov
        recur=intial-(0.2*solve(COVMATRIX)%*%t(SCORE))
        befor=intial
        intial=recur
        if(mean(abs(befor-recur))<0.00001){break}
      }
      filter[1,]=recur
      filter.var[1,,]=solve(-COVMATRIX)


      lihood=0
      for(j in 1:length(t[[1]])){
        y.hatt=exp(ZZ[,j]%*%filter[1,])/(1+exp(ZZ[,j]%*%filter[1,]))
        Covmatrix=-solve(prediction.var[1,,])-as.numeric(y.hatt*(1-y.hatt))*ZZ[,j]%o%ZZ[,j]
        densi=1/sqrt(det(prediction.var[1,,]))*
          exp(-0.5*(filter[1,]-prediction[1,])%*%solve(prediction.var[1,,])%*%(filter[1,]-prediction[1,]))
        lihood=lihood+log(2*pi*sqrt(det(solve(-Covmatrix)))*(y.hatt^(y[[1]][j]))*((1-y.hatt)^(1-y[[1]][j]))*densi)
      }
      Sample.likehood[1]=lihood


      for(i in 2:n.train) {
        prediction[i,]=TT%*%filter[i-1,]
        prediction.var[i,,]=TT%*%filter.var[i-1,,]%*%t(TT)+Q

        intial=prediction[i,]


        ZZ=matrix(0,dim*2+dim.con,length(t[[i]]))
        ZZ[1,]=1
        if(num.vary>0){
          for (sss in 1:(dim-1)) {
            ZZ[2*sss+1,]=t(X[[i]])[sss,]
          }
          ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[i]])
        }
        if(num.vary==0){
          ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[i]])
        }

        for(inter in 1:1000){
          summa=rep(0,dim*2+dim.con)
          summa.cov=matrix(0,dim*2+dim.con,dim*2+dim.con)
          for(j in 1:length(t[[i]])){
            y.hat=exp(ZZ[,j]%*%intial)/(1+exp(ZZ[,j]%*%intial))
            summa=summa+as.numeric(y[[i]][j]-y.hat)*ZZ[,j]
            summa.cov=summa.cov+as.numeric(y.hat*(1-y.hat))*ZZ[,j]%o%ZZ[,j]
          }
          SCORE=summa-t(solve(prediction.var[i,,])%*%(intial-matrix(prediction[i,],2*dim+dim.con,1)))
          COVMATRIX=-solve(prediction.var[i,,])-summa.cov
          recur=intial-(0.2*solve(COVMATRIX)%*%t(SCORE))
          befor=intial
          intial=recur
          if(max(abs(befor-recur))<0.00001){break}
        }
        filter[i,]=recur
        filter.var[i,,]=solve(-COVMATRIX)

        lihood=0
        for(j in 1:length(t[[i]])){
          y.hatt=exp(ZZ[,j]%*%filter[i,])/(1+exp(ZZ[,j]%*%filter[i,]))
          Covmatrix=-solve(prediction.var[i,,])-as.numeric(y.hatt*(1-y.hatt))*ZZ[,j]%o%ZZ[,j]
          densi=1/sqrt(det(prediction.var[i,,]))*
            exp(-0.5*(filter[i,]-prediction[i,])%*%solve(prediction.var[i,,])%*%(filter[i,]-prediction[i,]))
          lihood=lihood+log(2*pi*sqrt(det(solve(-Covmatrix)))*(y.hatt^(y[[i]][j]))*((1-y.hatt)^(1-y[[i]][j]))*densi)
        }
        Sample.likehood[i]=lihood
      }
      likelihood=-sum((Sample.likehood))
      return(likelihood)
    }

    if(dim>1){
      search.opt=optim(rep(2,dim),likehood.predictive)  #Optimize likelihood function with initial value (0,0)
      Lambda=exp(search.opt$par)
    }
    if(dim==1){
      search.opt=optimize(likehood.predictive,c(-20,20)) #Optimize likelihood function with initial value (0,0)
      Lambda=exp(search.opt$minimum)
    }
    #Smoothing parameter for intercept
  }else{
    if(length(Lambda)!=num.vary+1){
      stop("The number of smoothing parameters should equal to number of varying coefficients, including varying intercept.")
    }
    Lambda=Lambda
  }
  P1<-exp(diff)*diag(dim*2+dim.con)                   #Diffuse prior
  Q=matrix(0,dim*2+dim.con,dim*2+dim.con)

  Q1=matrix(0,2*dim,2*dim)
  for(ss in 1:dim){
    Q1[(2*ss-1):(2*ss),(2*ss-1):(2*ss)]=Lambda[ss]*QQ
  }
  Q2=TTq2=diag(rep(0,dim.con))
  Q=as.matrix(bdiag(Q1,Q2))

  #########################################
  #Model estimation using all data with Kalman filter
  #########################################

  prediction=matrix(NA,S0,2*dim+dim.con)                    # Mean prediction
  prediction.var=array(NA,dim=c(S0,2*dim+dim.con,2*dim+dim.con))  # Covariance prediction
  prediction[1,]=a1                                  # Initial mean
  prediction.var[1,,]=P1                             # Initial covariance

  filter=matrix(NA,S0,2*dim+dim.con)                        # Filtering mean
  filter.var=array(NA,dim=c(S0,2*dim+dim.con,2*dim+dim.con))      # Filtering covariance

  #Kalman filter on first timepoint
  intial=prediction[1,]
  ZZ=matrix(0,2*dim+dim.con,length(t[[1]]))  #define covariate z in our state space model (5)
  ZZ[1,]=1
  if(num.vary>0){
    for (sss in 1:(dim-1)) {
      ZZ[2*sss+1,]=t(X[[1]])[sss,]
    }
    ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[1]])
  }
  if(num.vary==0){
    ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[1]])
  }
  for(inter in 1:1000){       # Newton-Raphson iterative algorithm to get filtering estimator
    summa=rep(0,2*dim+dim.con)
    summa.cov=matrix(0,2*dim+dim.con,2*dim+dim.con)
    for(j in 1:length(t[[1]])){
      y.hat=exp(ZZ[,j]%*%intial)/(1+exp(ZZ[,j]%*%intial))
      summa=summa+as.numeric(y[[1]][j]-y.hat)*ZZ[,j]
      summa.cov=summa.cov+as.numeric(y.hat*(1-y.hat))*ZZ[,j]%o%ZZ[,j]
    }
    SCORE=summa-t(solve(prediction.var[1,,])%*%(intial-matrix(prediction[1,],2*dim+dim.con,1)))
    COVMATRIX=-solve(prediction.var[1,,])-summa.cov
    recur=intial-(0.2*solve(COVMATRIX)%*%t(SCORE))
    befor=intial
    intial=recur
    if(mean(abs(befor-recur))<0.00001){break}
    if(inter==1000){stop("failed converge","\n")}
  }
  filter[1,]=recur                      # Filtering mean
  filter.var[1,,]=solve(-COVMATRIX)     # Filtering covariance

  #Kalman filter on timepoints from 2 to S
  Prob.est=list()
  for(i in 2:S0) {
    # Prediction step
    prediction[i,]=TT%*%filter[i-1,]
    prediction.var[i,,]=TT%*%filter.var[i-1,,]%*%t(TT)+Q

    # Filtering step
    intial=prediction[i,]
    ZZ=matrix(0,dim*2+dim.con,length(t[[i]]))
    ZZ[1,]=1
    if(num.vary>0){
      for (sss in 1:(dim-1)) {
        ZZ[2*sss+1,]=t(X[[i]])[sss,]
      }
      ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[i]])
    }
    if(num.vary==0){
      ZZ[(2*dim+1):(2*dim+dim.con),]=t(XX[[i]])
    }
    if(i>n.train){
      y.hat=exp(t(ZZ)%*%intial)/(1+exp(t(ZZ)%*%intial))
      Prob.est[[i-n.train]]=y.hat
    }

    for(inter in 1:1000){     # Newton-Raphson
      summa=rep(0,dim*2+dim.con)
      summa.cov=matrix(0,dim*2+dim.con,dim*2+dim.con)
      for(j in 1:length(t[[i]])){
        y.hat=exp(ZZ[,j]%*%intial)/(1+exp(ZZ[,j]%*%intial))
        summa=summa+as.numeric(y[[i]][j]-y.hat)*ZZ[,j]
        summa.cov=summa.cov+as.numeric(y.hat*(1-y.hat))*ZZ[,j]%o%ZZ[,j]
      }
      SCORE=summa-t(solve(prediction.var[i,,])%*%(intial-matrix(prediction[i,],2*dim+dim.con,1)))
      COVMATRIX=-solve(prediction.var[i,,])-summa.cov
      recur=intial-(0.2*solve(COVMATRIX)%*%t(SCORE))
      befor=intial
      intial=recur
      if(mean(abs(befor-recur))<0.00001){break}
      if(inter==1000){stop("Newton-Raphson failed to converge","\n")}
      #if(inter==1000){break}
    }
    filter[i,]=recur
    filter.var[i,,]=solve(-COVMATRIX)
  }

  #Smoothed coefficients on training data
  state.smooth=matrix(NA,S0,2*dim+dim.con)
  F.t.smooth=array(NA,dim=c(S0,2*dim+dim.con,2*dim+dim.con))
  state.smooth[S0,]=filter[S0,]
  F.t.smooth[S0,,]=filter.var[S0,,]
  for(i in (S0-1):1){
    A.i=filter.var[i,,]%*%t(TT)%*%solve(prediction.var[i+1,,])
    state.smooth[i,]=filter[i,]+A.i%*%(state.smooth[i+1,]-prediction[i+1,])
    F.t.smooth[i,,]=filter.var[i,,]-A.i%*%(prediction.var[i+1,,]-F.t.smooth[i+1,,])%*%t(A.i)
  }
  Est=list(Pred=prediction,Pred.var=prediction.var,Filter=filter
           ,Filter.var=filter.var,Smooth=state.smooth,Smooth.var=F.t.smooth
           ,Lambda=Lambda,vary=vary,q=q,q1=q1,q2=q2
           ,dim=dim,dim.con=dim.con,TT=TT,Q=Q
           ,Prob.est=as.vector(Prob.est),S=S0,gap.len=delta.t)
  return(Est)
}

#' Prediction step of Kalman Filter
#' @author Jiakun Jiang, Wei Yang, Stephen E. Kimmel and Wensheng Guo
#' @description Given the estimates of the smoothing parameters, the K-step-ahead prediction can be done by running the Kalman filtering prediction steps without the filtering steps.
#' @param fit A object generated by function DLSSM() or DLSSM.init() or DLSSM.filter()
#' @param newx Covariates matrix of subjects if you are interested in predicting their event probabilities in the future.
#' @param K K-steps ahead of prediction
#' @export
#' @return List of predicted coefficients and probabilities if covariates are provided.
#'  \tabular{ll}{
#'    \code{coef.pred} \tab Matrix with row dimension K, including one-step ahead to K-steps ahead coefficients prediction. The k-th row correspond to k-step ahead prediction. \cr
#'    \tab \cr
#'    \code{coef.pred.var} \tab Array of covariance matrix corresponding to  coef.pred. \cr
#'    \tab \cr
#'    \code{prob.pred} \tab Matrix with column dimension K, including predicted probabilities of subjects if covariates are provided. The k-th column correspond to k-step ahead prediction.  \cr
#'  }
DLSSM.predict<-function(fit,newx,K){
  vary=fit$vary
  num.vary=length(vary)
  dim=fit$dim
  dim.con=fit$dim.con
  if(is.vector(fit$Filter)==T){
    fil.last=fit$Filter
    filt.var.last=fit$Filter.var
  }
  else{
    fil.last=fit$Filter[dim(fit$Filter)[1],]
    filt.var.last=fit$Filter.var[dim(fit$Filter)[1],,]
  }
  coef.prediction=matrix(NA,K,2*dim+dim.con)                    # Mean prediction
  coef.prediction.var=array(NA,dim=c(K,2*dim+dim.con,2*dim+dim.con))  # Covariance prediction
  for(pred in 1:K){
    TT=fit$TT
    Q=fit$Q
    coef.prediction[pred,]=TT%*%fil.last
    coef.prediction.var[pred,,]=TT%*%filt.var.last%*%t(TT)+Q
  }
  if(is.null(newx)==TRUE){prob.pred=NULL}
  if(is.null(newx)==FALSE){
    if(is.vector(newx)==T) {newx=matrix(newx,nrow=1)}
    if(dim(newx)[2]!=fit$q) stop("The dimension of covariates does not match that dimension in the fitted model")
    prob.pred=matrix(NA,dim(newx)[1],K)
    for(kkk in 1:K){
      prediction=coef.prediction[kkk,]
      ZZ=matrix(0,dim(newx)[1],dim*2+dim.con)
      ZZ[,1]=1
      for(sss in 1:(dim-1)){
        ZZ[,2*sss+1]=newx[,sss]
      }
      ZZ[,(2*dim+1):(2*dim+dim.con)]=newx[,dim:fit$q]
      prob.pred[,kkk]=exp(ZZ%*%prediction)/(1+exp(ZZ%*%(prediction)))
    }
  }
  return(list(coef.pred=coef.prediction,coef.pred.var=coef.prediction.var,prob.pred=prob.pred))
}

#' Filtering step of Kalman Filter
#' @author Jiakun Jiang, Wei Yang, Stephen E. Kimmel and Wensheng Guo
#' @description When the new data becomes available, the coefficients can be efficiently
#'  updated by running filtering step from the original estimates over the new observations.
#' @param fit A object generated by last step operation, could be DLSSM(), DLSSM.init() or DLSSM.filter().
#' @param newx Covariates matrix of new batch data
#' @param newy Binary outcome of new batch data
#' @export
#' @return See the returned value of DLSSM().
DLSSM.filter<-function(fit,newx,newy){
  vary=fit$vary
  num.vary=length(vary)
  dim=fit$dim
  dim.con=fit$dim.con
  TT=fit$TT
  Q=fit$Q
  q=fit$q
  if(is.vector(fit$Filter)==T){
    fil.last=fit$Filter
    filt.var.last=fit$Filter.var
  }
  else{
    fil.last=fit$Filter[dim(fit$Filter)[1],]
    filt.var.last=fit$Filter.var[dim(fit$Filter)[1],,]
  }

  predict=TT%*%fil.last
  predict.var=TT%*%filt.var.last%*%t(TT)+Q

  # Filtering step
  intial=predict
  ZZ=matrix(0,2*dim+dim.con,length(newy))
  ZZ[1,]=1
  if(num.vary>0){
    for (sss in 1:(dim-1)) {
      ZZ[2*sss+1,]=t(newx)[sss,]
    }
    ZZ[(2*dim+1):(2*dim+dim.con),]=t(newx)[dim:q,]
  }
  if(num.vary==0){
    ZZ[(2*dim+1):(2*dim+dim.con),]=t(newx)
  }

  for(inter in 1:1000){     # Newton-Raphson
    summa=rep(0,dim*2+dim.con)
    summa.cov=matrix(0,dim*2+dim.con,dim*2+dim.con)
    for(j in 1:length(newy)){
      y.hat=exp(ZZ[,j]%*%intial)/(1+exp(ZZ[,j]%*%intial))
      summa=summa+as.numeric(newy[j]-y.hat)*ZZ[,j]
      summa.cov=summa.cov+as.numeric(y.hat*(1-y.hat))*ZZ[,j]%o%ZZ[,j]
    }
    SCORE=summa-t(solve(predict.var)%*%(intial-matrix(predict,2*dim+dim.con,1)))
    COVMATRIX=-solve(predict.var)-summa.cov
    recur=intial-(0.2*solve(COVMATRIX)%*%t(SCORE))
    befor=intial
    intial=recur
    if(mean(abs(befor-recur))<0.00001){break}
    if(inter==1000){stop("failed converge","\n")}
  }
  filter=as.vector(recur)
  filter.var=solve(-COVMATRIX)
  Lambda=fit$Lambda
  q=fit$q
  q1=fit$q1
  Est=list(Filter=filter,Filter.var=filter.var,Lambda=Lambda,vary=vary,q=q,q1=q1,dim=dim,dim.con=dim.con,TT=TT,Q=Q)
  return(Est)
}

#' Smoothing step in state-space model
#' @author Jiakun Jiang, Wei Yang, Stephen E. Kimmel and Wensheng Guo
#' @description Smoothing past estimated coefficients.
#' @param fit A object generated by function DLSSM() or DLSSM.init() or DLSSM.filter()
#' @param prediction Matrix of all prediction coefficients
#' @param prediction.var Array of all prediction covariance matrix
#' @param filter Matrix of all filtering coefficients
#' @param filter.var Array of all filtering covariance matrix
#' @export
#' @return Smoothed coefficients and corresponding covariance matrix.
#'  \tabular{ll}{
#'    \code{smooth} \tab Smooothed coefficients. \cr
#'    \tab \cr
#'    \code{smooth.var} \tab Array of covariance matrix corresponding to smooth. \cr
#'  }
DLSSM.smooth<-function(fit,prediction,prediction.var,filter,filter.var){
  dim=fit$dim
  dim.con=fit$dim.con
  TT=fit$TT
  n.train=dim(filter)[1]
  state.smooth=matrix(NA,n.train,2*dim+dim.con)
  F.t.smooth=array(NA,dim=c(n.train,2*dim+dim.con,2*dim+dim.con))
  state.smooth[n.train,]=filter[n.train,]
  F.t.smooth[n.train,,]=filter.var[n.train,,]
  for(i in (n.train-1):1){
    A.i=filter.var[i,,]%*%t(TT)%*%solve(prediction.var[i+1,,])
    state.smooth[i,]=filter[i,]+A.i%*%(state.smooth[i+1,]-prediction[i+1,])
    F.t.smooth[i,,]=filter.var[i,,]-A.i%*%(prediction.var[i+1,,]-F.t.smooth[i+1,,])%*%t(A.i)
  }
  return(list(smooth=state.smooth,smooth.var=F.t.smooth))
}

#' Dynamic logistic state-space prediction model for binary outcomes
#' @author Jiakun Jiang, Wei Yang, Stephen E. Kimmel and Wensheng Guo
#' @description Plot predicted, filtered and smoothed coefficients
#' @param fit fitted state-space model
#' @export
#' @details The argument fit could be object of DLSSM or DLSSM.init
DLSSM.plot<-function(fit){
  S=fit$S
  q2=fit$q2
  rows=length(fit$vary)+1
  training_samp=fit$training_samp
  cc=rows+q2
  c1=ceiling(sqrt(cc))
  f1=floor(sqrt(cc))
  if(cc==c1*f1){numb.plot=c(f1,c1)}
  if(c1*f1>cc){numb.plot=c(f1,c1)}
  if(c1*f1<cc){numb.plot=c(c1,c1)}
  plot.index=c(2*(1:rows)-1,(2*rows+1):(2*rows+q2))
  par(mfrow=numb.plot,mar=c(4, 4, 1, 1),oma=c(1,1,1,1))

  est.p=fit$Pred
  est.var.p=fit$Pred.var
  est.f=fit$Filter
  est.var.f=fit$Filter.var
  est.s=fit$Smooth
  est.var.s=fit$Smooth.var

  for(ss in plot.index){
    f.up1=est.f[,ss]+2*sqrt(est.var.f[,ss,ss])
    f.low1=est.f[,ss]-2*sqrt(est.var.f[,ss,ss])
    p.up1=est.p[,ss]+2*sqrt(est.var.p[,ss,ss])
    p.low1=est.p[,ss]-2*sqrt(est.var.p[,ss,ss])
    s.up1=est.s[,ss]+2*sqrt(est.var.s[,ss,ss])
    s.low1=est.s[,ss]-2*sqrt(est.var.s[,ss,ss])
    p_v_up1=p.up1#[(training_samp+1):S]
    p_v_low1=p.low1#[(training_samp+1):S]
    f_v_up1=f.up1#[1:training_samp]
    f_v_low1=f.low1#[1:training_samp]
    smoo_v_up1=s.up1#[1:training_samp]
    smoo_v_low1=s.low1#[1:training_samp]
    if(ss %in% (2*(1:rows)-1)){
      lowbound=min(c(p_v_low1)[-c(1,2)])-0.25*diff(range(c(p_v_low1)[-c(1,2)]))
      maxbound=max(c(p_v_up1)[-c(1,2)])+0.25*diff(range(c(p_v_up1)[-c(1,2)]))
      plot(NULL,xlim=c(0,1),ylim=c(lowbound,maxbound),xlab="t",ylab=bquote(beta[.((ss+1)/2-1)]~(t)))
    }
    if(ss %in% (2*rows+1):(2*rows+q2)){
      lowbound=min(c(p_v_low1)[-c(1,2)])-0.25*max(diff(range(c(p_v_low1)[-c(1,2)])),0.2)
      maxbound=max(c(p_v_up1)[-c(1,2)])+0.25*max(diff(range(c(p_v_up1)[-c(1,2)])),0.2)
      plot(NULL,xlim=c(0,1),ylim=c(lowbound,maxbound),xlab="t",ylab=bquote(alpha[.(ss-2*rows)]))
    }
    pre=est.p[,ss]
    fil=est.f[,ss]
    smoo=est.s[,ss]
    loc1=c(1:S)*fit$gap.len
    loc2=c(1:S)*fit$gap.len
    lines(loc2,pre,lty=1,col="blue")
    lines(loc2,p_v_up1,lty=3,col="blue")
    lines(loc2,p_v_low1,lty=3,col="blue")
    lines(loc1,fil,lty=1)
    lines(loc1,f_v_up1,lty=3)
    lines(loc1,f_v_low1,lty=3)
    lines(loc1,smoo,lty=1,col="red")
    lines(loc1,smoo_v_up1,lty=3,col="red")
    lines(loc1,smoo_v_low1,lty=3,col="red")
    vv=fit$gap.len*training_samp
    abline(v=vv,col="grey",lty=2)
    legend('bottomright',c("predict","filter","smooth"),lty=c(1,1,1),col=c("blue","black","red"),text.width=0.15,seg.len=0.5)
  }
}


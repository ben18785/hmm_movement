source(file = "Coords_to_ThetaGamma.R")
require(readr)
require(TMB)
greyseal <- read_csv("../data/greyseal.csv")

compile(file="./Cpp_code/Multistate_theta_gamma_HMM/Multistate_theta_gamma_HMM.cpp")
compile(file="./Cpp_code/Theta_mixture/Theta_mixture.cpp")
dyn.load(dynlib("./Cpp_code/Multistate_theta_gamma_HMM/Multistate_theta_gamma_HMM"))
dyn.load(dynlib("./Cpp_code/Theta_mixture/Theta_mixture"))


###
###  Change Seal locations to deflection angles and gammes
###
{
delta = 3*60*60
t0 = greyseal$date[[1]]
tT = greyseal$date[length(greyseal$date)]
t = seq(t0, tT, by=delta)
sealLoc <- cbind(approx(greyseal$date, greyseal$lon, xout = t)$y,
                 approx(greyseal$date, greyseal$lat, xout = t)$y)
colnames(sealLoc)<- c("Lon","Lat")
sealLoc<- as.data.frame(sealLoc)

gamma.series<- sapply(2:(nrow(sealLoc)-1),
                      function(t) distance.formula(x.last = sealLoc[t,],
                                                   x.curr = sealLoc[t+1,],
                                                   lat = 2,
                                                   lon = 1)/
                        distance.formula(x.last = sealLoc[t-1,],
                                         x.curr = sealLoc[t,],
                                         lat = 2,
                                         lon = 1))

angle.series<- sapply(2:(nrow(sealLoc)-1), function(t) deflection.angle.formula(x.last = sealLoc[t-1,],
                                                                                x.curr = sealLoc[t,],
                                                                                x.next = sealLoc[t+1,],
                                                                                lat = 2,
                                                                                lon = 1))
}

###
###  Plot the data
###
{
  par(mfrow=c(2,1))
    acf(log(gamma.series),
        xlab = "ACF for log(gamma)",
        main = "")
    hist(log(gamma.series),
         breaks = 300,
         xlab = "Histogram for log(gamma)",
         main = "")
    
    acf(angle.series,
        xlab = "ACF for theta",
        main = "")
    hist(c(angle.series-360,angle.series,angle.series+360),
         breaks = 450,
         xlab = "Histogram for theta",
         main = "")
  par(mfrow=c(1,1))
}

###
###  Run TMB Analysis
###
map<- FALSE          ### If map <- TRUE, then TMB will fix the HMM distribution parameters as those found by the mixture model
n_states<- 2           ### Can set this to whatever you want it to be
{
  thetas<- seq((n_states-1)*pi/n_states,       ### Evenly spaced around circle
               2*pi,
               2*pi/n_states)
  logit_probs <- matrix(runif(n_states-1,      ### Random starting values to run to the mixture model, state probabilities
                              min = -1,
                              max = 1),
                        nrow = 1)
  wc_logit_pars<- as.matrix(cbind(thetas,      ### Random starting values to run to the mixture model, state parameters
                                  runif(n_states,
                                        min = -2,
                                        max = 2)))
  
  
  mix.ADFun<- MakeADFun(data = list(theta = pi/180*angle.series),     ### Run the mixture model
                        parameters = list(logit_probs = logit_probs,
                                          wc_logit_pars = wc_logit_pars),
                        DLL = "Theta_mixture",
                        silent = TRUE)
  mix.optim<- nlminb(mix.ADFun$par,        ### Get ML Estimates for mixture model
                     mix.ADFun$fn,
                     mix.ADFun$gr,
                     control = list(trace = 0))
  
  
  
  beta_matrix<- matrix(runif(n_states*(2*n_states-2),    ### Random starting values to run to the full HMM, state transition probabilities
                             min=-3,
                             max=3),
                       nrow = n_states,
                       ncol = 2*n_states-2)
  wc_logit_pars<- mix.ADFun$report()$wc_logit_pars       ### Use the mixture model parameters as starting values for full HMM
  
  
  
  if( map == TRUE ){
    ADFun<- MakeADFun(data = list(theta = pi/180*angle.series,      ### Run the full HMM
                                  gamma = gamma.series),
                      parameters = list(beta_matrix = beta_matrix,
                                        wc_logit_pars = wc_logit_pars,
                                        logSigma = 0),
                      map = list(wc_logit_pars = factor(matrix(NA,nrow = n_states,ncol = 2)) ),
                      DLL = "Multistate_theta_gamma_HMM",
                      silent = TRUE)
  } else {
    ADFun<- MakeADFun(data = list(theta = pi/180*angle.series,      ### Run the full HMM
                                  gamma = gamma.series),
                      parameters = list(beta_matrix = beta_matrix,
                                        wc_logit_pars = wc_logit_pars,
                                        logSigma = 0),
                      DLL = "Multistate_theta_gamma_HMM",
                      silent = TRUE)
  }
  optim<- nlminb(ADFun$par,           ### Get ML Estimates for mixture model
                 ADFun$fn,
                 ADFun$gr,
                 control = list(maxit = 10000))
  
  mix.sd<- sdreport(mix.ADFun)    ### Get standard errors
  sd<-sdreport(ADFun)
  
  cat(c(paste0(" Mixture model convergence:       ",mix.optim$message,'\n'),    ### Check convergence of models
        paste0("Hidden Markov model convergence: ",optim$message)))
}

mix.ADFun$report()
ADFun$report()

summary(mix.sd, p.value = T)        ### Values of matrices are reported by concatenating successive columns
summary(sd, p.value = T)     

###
###  Plot the Marginal Distribution and check against histogram
###
{
  dwrapcauchy<- function(theta, mu, rho){         ### Pdf used to fit the model
    return(1/(2*pi) * (1 - rho^2)/(1 + rho^2 - 2*rho*cos(theta - mu)))
  }
  x<- 3*seq(-180,180,0.1)
  
  pars<- mix.ADFun$report()$wc_pars
  
  prob<- sapply(1:n_states, function(x) {    ### Get the marginal probability for state occurence using the Viterbi path
    states<- sum(ADFun$report()$viterbi_path==x)
    total<- length(ADFun$report()$viterbi_path==x)
    return(states/total)
  })
  
  marginal.dist<- function(x) {     ### Marginal distribution is the sum of the probability of each state times the pdf of x at that state
    sapply(x, function(y) {
      sum(sapply(1:n_states, function(i) {
        return(prob[[i]]*dwrapcauchy(theta = y,
                                     mu = pars[i,1],
                                     rho = pars[i,2]))
      }))
    })
  }
  
  par(mfrow=c(2,1))
  hist(c(angle.series-360,angle.series,angle.series+360),       ### Plot the data
       breaks = 3*180,
       freq = TRUE,
       xlab = "Theta (deg.)",
       ylab = "Freq.",
       main = "Data for Theta")
  plot(x,                                    ###  Plot the marginal distribution
       marginal.dist(pi/180*x),
       type = "l",
       xlab = "Theta (deg.)",
       ylab = "f(x)",
       main = "Marginal and State Distributions for Theta")
  for(i in 1:n_states) {                     ### Plot the state contributions to the marginal distribution
    lines(x,
          prob[[i]]*dwrapcauchy(theta = pi/180*x,
                                mu = pars[i,1],
                                rho = pars[i,2]),
          lty = i+1)
  }
  par(mfrow=c(1,1))
}

###
###  Plot the Switching Probabilities
###
begin<- 1
end<- length(gamma.series)
gamma.subset<- gamma.series[begin:end]
{
  betas<- ADFun$report()$beta_matrix
  
  par(mfrow=c(n_states,n_states))
  for(i in 1:n_states){
    for(j in 1:n_states){
      if(i == j) {
        plot.ts(log(gamma.subset),
                main="",
                xlab="log(gamma)",
                ylab="")
      } else if( i<j ) {
        prob<- 1/(1 + exp(betas[i,2*j-3] + betas[i,2*j-2]*log(gamma.subset)))
        plot.ts(prob,
                main="",
                xlab=paste("State",i,"to state",j),
                ylab="")
      } else if( i>j ) {
        prob<- 1/(1 + exp(betas[i,2*j-1] + betas[i,2*j]*log(gamma.subset)))
        plot.ts(prob,
                main="",
                xlab=paste("State",i,"to state",j),
                ylab="")
      }
    }
  }
  par(mfrow=c(1,1))
}
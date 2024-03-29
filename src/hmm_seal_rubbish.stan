functions{
  // function that returns the log pdf of the wrapped-Cauchy
  real wrapped_cauchy_lpdf(real phi, real rho, real nu) {
    return(- log(2 * pi()) + log((1 - rho^2)/(1 + rho^2 - 2 * rho * cos(phi - nu))));
  }
}

data {
  int<lower=0> N;
  vector[N] dist;
  vector<lower=-(pi()+0.01), upper=(pi()+0.01)>[N] angle;
  int<lower=1> K;
  real<lower=0> cost;
  real<lower=0, upper=1> p_anomalous;
  real<lower=1> conc_anomalous;
}

transformed data {
  real epsilon = 1e-6;
}

parameters {
  simplex[K] theta[K];
  vector<lower=0>[K] sigma;
  // wrapped Cauchy params
  vector<lower=-pi(), upper=pi()>[K] nu;
  vector<lower=0,upper=1>[K] rho;
  real<lower=0, upper=1> phi;
  vector<lower=0>[K] a;
  vector<lower=0>[K] b;
}

transformed parameters {
  vector[K + 1] log_theta_tr[K + 1];
  vector[K + 1] lp;
  vector[K + 1] lp_p1;
  simplex[K + 1] theta_1[K + 1];
  
  for(k_from in 1:(K + 1)) {
    for(k_to in 1:(K + 1)) {
      if(k_from <= K) { // not going from lost state
        if(k_to <= K) {
          theta_1[k_from, k_to] = (1 - phi) * theta[k_from, k_to];
        } else {
          theta_1[k_from, k_to] = phi;
        }
      } else { // from lost state
        if(k_to <= K) {
          theta_1[k_from, k_to] = (1 - phi) / 2;
        } else {
          theta_1[k_from, k_to] = phi;
        }
      }
    }
  }
  
  
  lp = log1m(phi) + rep_vector(-log(K), K + 1);
  lp[K + 1] = log(phi);
  
   for (k_from in 1:(K + 1))
    for (k in 1:(K + 1))
      log_theta_tr[k, k_from] = log(theta_1[k_from, k]);
      
  // Forwards algorithm
  for (n in 1:N) {
    for (k in 1:(K + 1)){
      if(k <= K) {
        lp_p1[k]
          = log_sum_exp(log_theta_tr[k] + lp)
          + wrapped_cauchy_lpdf(angle[n] | rho[k], nu[k])
          + gamma_lpdf(dist[n] + epsilon | a[k], b[k]);
      } else {
        lp_p1[k] = log_sum_exp(log_theta_tr[k] + lp) - cost;
      }
    }
    lp = lp_p1;
  }
}
model {
  target += log_sum_exp(lp);
  phi ~ beta(conc_anomalous * p_anomalous, conc_anomalous * (1 - p_anomalous));
  a ~ cauchy(0, 5);
  b ~ cauchy(0, 5);
}


generated quantities {
  int<lower=1,upper=K+1> state[N];
  real log_p = log_sum_exp(lp);
  real log_p_y_star;
  {
    int back_ptr[N, K + 1];
    real best_logp[N, K + 1];
    real best_total_logp;
    for (k in 1:K)
      best_logp[1, k] = gamma_lpdf(dist[1] + epsilon | a[k], b[k]) +
        wrapped_cauchy_lpdf(angle[1] | rho[k], nu[k]);
    best_logp[1, K + 1] = phi;
    for (t in 2:N) {
      for (k in 1:(K + 1)) {
      best_logp[t, k] = negative_infinity();
        for (j in 1:(K + 1)) {
          real logp;
          logp = best_logp[t-1, j] + log(theta_1[j, k]);
          if(k <= K) {
            logp += gamma_lpdf(dist[t] + epsilon | a[k], b[k])
            + wrapped_cauchy_lpdf(angle[t] | rho[k], nu[k]);
          } else {
            logp -= cost;
          }
          if (logp > best_logp[t, k]) {
            back_ptr[t, k] = j;
            best_logp[t, k] = logp;
          }
        }
      }
    }
    log_p_y_star = max(best_logp[N]);
    for (k in 1:(K + 1))
      if (best_logp[N, k] == log_p_y_star)
        state[N] = k;
      for (t in 1:(N - 1))
        state[N - t] = back_ptr[N - t + 1, state[N - t + 1]];
  }
}

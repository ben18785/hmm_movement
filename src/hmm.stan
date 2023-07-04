data {
  int<lower=0> N;
  vector[N] dist;
  int<lower=1> K;
}
parameters {
  simplex[K] theta[K];
  positive_ordered[K] mu;
  vector<lower=0>[K] sigma;
}
transformed parameters {
  vector[K] log_theta_tr[K];
  vector[K] lp;
  vector[K] lp_p1;
  
  lp = rep_vector(-log(K), K);
  
   for (k_from in 1:K)
    for (k in 1:K)
      log_theta_tr[k, k_from] = log(theta[k_from, k]);
      
  // Forwards algorithm
  for (n in 1:N) {
    for (k in 1:K){
      lp_p1[k]
        = log_sum_exp(log_theta_tr[k] + lp)
        + normal_lpdf(dist[n] | mu[k], sigma[k]);
    }
    lp = lp_p1;
  }
}
model {
  target += log_sum_exp(lp);
}


generated quantities {
  int<lower=1,upper=K> state[N];
  real log_p = log_sum_exp(lp);
  real log_p_y_star;
  {
    int back_ptr[N, K];
    real best_logp[N, K];
    real best_total_logp;
    for (k in 1:K)
      best_logp[1, K] = normal_lpdf(dist[1] | mu[k], sigma[k]);
    for (t in 2:N) {
      for (k in 1:K) {
      best_logp[t, k] = negative_infinity();
        for (j in 1:K) {
          real logp;
          logp = best_logp[t-1, j]
          + log(theta[j, k]) + normal_lpdf(dist[t] | mu[k], sigma[k]);
          if (logp > best_logp[t, k]) {
            back_ptr[t, k] = j;
            best_logp[t, k] = logp;
          }
        }
      }
    }
    log_p_y_star = max(best_logp[N]);
    for (k in 1:K)
      if (best_logp[N, k] == log_p_y_star)
      state[N] = k;
      for (t in 1:(N - 1))
      state[N - t] = back_ptr[N - t + 1,
      state[N - t + 1]];
  }
}

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
  vector<lower=0>[K] a;
  vector<lower=0>[K] b;
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
        + gamma_lpdf(dist[n] + epsilon | a[k], b[k])
        + wrapped_cauchy_lpdf(angle[n] | rho[k], nu[k]);
    }
    lp = lp_p1;
  }
}
model {
  target += log_sum_exp(lp);
  a ~ cauchy(0, 5);
  b ~ cauchy(0, 5);
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
      best_logp[1, K] = gamma_lpdf(dist[1] + epsilon | a[k], b[k]) +
      wrapped_cauchy_lpdf(angle[1] | rho[k], nu[k]);
    for (t in 2:N) {
      for (k in 1:K) {
      best_logp[t, k] = negative_infinity();
        for (j in 1:K) {
          real logp = best_logp[t-1, j]
            + log(theta[j, k])
            + gamma_lpdf(dist[t] + epsilon | a[k], b[k])
            + wrapped_cauchy_lpdf(angle[t] | rho[k], nu[k]);
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
        state[N - t] = back_ptr[N - t + 1, state[N - t + 1]];
  }
}

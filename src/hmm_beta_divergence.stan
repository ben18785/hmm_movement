functions {
  real normal_density(real x,
                    real xc,
                    real[] theta,
                    real[] x_r,
                    int[] x_i)  {
    real mu = theta[1];
    real sigma = theta[2];
    real beta = theta[3];
  
    return (1 / (sqrt(2 * pi()) * sigma) * exp(-0.5 * ((x - mu) / sigma)^2)) ^ (beta + 1);
  }
  
  real normal_int(real mu, real sigma, real beta, real[] x_r,
                    int[] x_i) {
    return(integrate_1d(normal_density, negative_infinity(), positive_infinity(), {mu, sigma, beta}, x_r, x_i, 0.0001));
  }
  
  real robust_normal_lpdf(real x, real mu, real sigma, real beta, real int_value) {
    return(1 / beta * exp(normal_lpdf(x | mu, sigma))^beta - (1 / (beta + 1)) * int_value);
  }
}

data {
  int<lower=0> N;
  vector[N] dist;
  int<lower=1> K;
  real beta;
}
transformed data {
  real x_r[0];
  int x_i[0];
}
parameters {
  simplex[K] theta[K];
  positive_ordered[K] mu;
  vector<lower=0>[K] sigma;
}
transformed parameters {
  vector[K] log_theta_tr[K];
  real int_value[K];
  vector[K] lp;
  vector[K] lp_p1;
  
  lp = rep_vector(-log(K), K);
  
   for (k_from in 1:K)
    for (k in 1:K)
      log_theta_tr[k, k_from] = log(theta[k_from, k]);
      
  for(i in 1:K)
    int_value[i] = normal_int(mu[i], sigma[i], beta, x_r, x_i);
      
  // Forwards algorithm
  for (n in 1:N) {
    for (k in 1:K){
      lp_p1[k]
        = log_sum_exp(log_theta_tr[k] + lp)
        + robust_normal_lpdf(dist[n] | mu[k], sigma[k], beta, int_value[k]);
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
      best_logp[1, K] = robust_normal_lpdf(dist[1] | mu[k], sigma[k], beta, int_value[k]);
    for (t in 2:N) {
      for (k in 1:K) {
      best_logp[t, k] = negative_infinity();
        for (j in 1:K) {
          real logp;
          logp = best_logp[t-1, j]
          + log(theta[j, k]) + robust_normal_lpdf(dist[t] | mu[k], sigma[k], beta, int_value[k]);
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

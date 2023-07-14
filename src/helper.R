library(rstan)
options(mc.cores=4)
rstan_options("auto_write" = TRUE)
library(tidyverse)

select_new_state <- function(transition_probs) {
  draw <- rmultinom(1, 1, transition_probs)
  which(draw==1)
}

simulate_hmm_states <- function(n_steps, state_initial, m_transition_probs) {
  states <- vector(length = n_steps)
  state <- state_initial
  for(i in 1:n_steps) {
    transition_probs <- m_transition_probs[state, ]
    state <- select_new_state(transition_probs)
    states[i] <- state
  }
  states
}

optimise_repeat <- function(n_opt, model, data_stan) {
  best_val <- -Inf
  for(i in 1:n_opt) {
    fit <- optimizing(model, data=data_stan, as_vector=FALSE)
    val <- fit$value
    if(val > best_val) {
      best_val <- val
      fit_best <- fit
    }
  }
  fit_best
}

r_student_t <- function(mu, sigma, df) {
  mu + sigma * rt(n = 1, df = df) #/ sqrt(df / (df - 2))
}

tainted_normal <- function(w, mu, sigma, df) {
  
  u <- runif(1)
  if(u < w)
    x <- r_student_t(mu, sigma, df)
  else
    x <- rnorm(1, mu, sigma)
  
  x
}

generate_emissions <- function(states, w, mus, sigmas, dfs) {
  xs <- vector(length = length(states))
  for(i in seq_along(xs))
    xs[i] <- tainted_normal(
      w, mus[states[i]], sigmas[states[i]], dfs[states[i]])
  
  xs
}

simulate_hmm <- function(n_steps,
                         state_initial,
                         m_transition_probs,
                         w, mus, sigmas, dfs) {
  
  states <- simulate_hmm_states(n_steps, state_initial, m_transition_probs)
  x <- generate_emissions(states, w, mus, sigmas, dfs)
  
  df <- tibble(state=states, obs=x) %>% 
    mutate(time=seq_along(state)) %>% 
    mutate(state=as.factor(state))
  
  df
}
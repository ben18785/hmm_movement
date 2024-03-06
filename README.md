# Fitting HMMs allowing for rubbish collection

This repo holds materials for fitting HMMs including so-called rubbish collection states. The two files that contain such analyses are: `src/s_hmm_bernoull.Rmd`, which works on simulated data and (at the end) on elephant data; and `src/seal_analysis.Rmd`, which works on Joanna's seal data. The analyses in these two files use `Stan` to code up the HMMs and `rstan` to fit the models to the data.

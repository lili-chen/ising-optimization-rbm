# Gibbs Sampling for MAX-CUT Approximation

A comparison of the performance of discrete and continuous (Restricted) Boltzmann Machines in approximating the combinatorially hard MAX-CUT problem for 10-vertex graphs up to 150-vertex graphs.

## Efficiency vs. Problem Size

Run write_monte_carlo_to_csv.py and subsequently read_monte_carlo_from_csv.ipynb to see plots of the performance of the samplers on each problem size. We run each sampler for a fixed amount of samples, and want to see how close it gets to the correct cut (calculate algorithm_cut/theoretical_max_cut for each problem and average over all problems for each problem size).

## Performance vs. Trial Number

Run performance_with_trial_number-[discrete/rbm_cont/rbm-discrete/continuous].ipynb for a closer look at each sampler, and each problem size for that sampler.

## Comparison of Samplers

Run sampler_comparison.ipynb to see a comparison of the graphs from performance_with_trial_number-[discrete/rbm_cont/rbm-discrete/continuous].ipynb across samplers. (Since trials are counted differently for the various samplers, you may need to adjust the num_trials passed into each sampler).

## Author

Lili Chen


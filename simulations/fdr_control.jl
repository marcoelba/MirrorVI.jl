# FDR definitions
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using StatsFuns
using StatsBase
using HypothesisTests
using DataFrames
using GLM
using GLMNet

abs_project_path = normpath(joinpath(@__FILE__, ".."))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))

results_path = joinpath(abs_project_path, "results", "ms_analysis")

# Frequentist FDR is defined as E(FDP)
# FDP = False Discovery Proportion

n_tests = 100
pi0 = 0.8
n_null = Int(n_tests * pi0)
n_true = n_tests - n_null
alpha = 0.1
fdr_level = 0.1
# sample size
truth_label = vcat(zeros(n_null), ones(n_true))

# BH method and mixture model for frequentist FDR control
p_values = rand(n_null)
append!(p_values, rand(Beta(0.1, 10), n_true)) # alternative hyp dist concentrated near 0 
histogram(p_values)

# Using the non-parametric BH correction
sum(p_values .< alpha) / n_tests
bh_pvalues = MirrorStatistic.bh_correction(p_values=p_values, fdr_level=0.1)
sum(bh_pvalues[:, 3])
sum(bh_pvalues[:, 3] .* (1 .- truth_label)) / sum(bh_pvalues[:, 3])

# assuming mixture model
t = 0.01
pi0 * t / (pi0 * t + (1 - pi0)*cdf(Beta(0.1, 10), t))
sum((p_values .<= t) .* (1 .- truth_label)) / sum((p_values .<= t))


# ----------------------------------------------------------
# for the regression
# Generate N samples from the n-dimensional population
mu_population = zeros(n_tests)
dist_population = MultivariateNormal(ones(n_tests))
beta_true = vcat(
    rand([-1, 1], n_true),
    zeros(n_null)
)

N = 500
Random.seed!(134)
X = rand(dist_population, N)'
# density(X[1:20, :]')

y = X * beta_true .+ randn(N)


# simple parametric regression
lm_result = lm(X, y)
lm_pvalues = coeftable(lm_result).cols[4]
sum(lm_pvalues .< alpha)

histogram(lm_pvalues)
# Uniform distribution of p-values under the null hypothesis
MirrorStatistic.false_discovery_rate(
    true_coef=beta_true .!= 0,
    estimated_coef=lm_pvalues .< alpha
)

# BH correction for FDR
bh_pvalues = MirrorStatistic.bh_correction(p_values=lm_pvalues, fdr_level=0.1)
sum(bh_pvalues[:, 3])
MirrorStatistic.false_discovery_rate(
    true_coef=beta_true .!= 0,
    estimated_coef=bh_pvalues[:, 3]
)

# --------------------------------------------------------------
# FDR is defined as the expectation over repeated sampling
# so, do repeated sampling
Random.seed!(134)
n_rep = 100
fdr_no_correction = []
fdr_with_correction = []

for rep = 1:n_rep

    Xr = rand(dist_population, N)'
    yr = Xr * beta_true .+ randn(N)

    # simple lm
    lm_result = lm(Xr, yr)
    lm_pvalues = coeftable(lm_result).cols[4]

    push!(
        fdr_no_correction,
        MirrorStatistic.false_discovery_rate(
            true_coef=beta_true .!= 0,
            estimated_coef=lm_pvalues .< alpha
        )
    )
    # BH correction for FDR
    bh_pvalues = MirrorStatistic.bh_correction(p_values=lm_pvalues, fdr_level=fdr_level)
    push!(
        fdr_with_correction,
        MirrorStatistic.false_discovery_rate(
            true_coef=beta_true .!= 0,
            estimated_coef=bh_pvalues[:, 3]
        )
    )
end

mean(fdr_no_correction)
mean(fdr_with_correction)
histogram(fdr_no_correction)
histogram(fdr_with_correction)


# ----------------------------------------------------
# Knockoffs method
X_k = rand(dist_population, N)'

X_a = hcat(X, X_k)

lm_result = GLMNet.glmnetcv(X_a, y)
lm_coefs = GLMNet.coef(lm_result)

knockoff_mirror_stat = abs.(lm_coefs[1:n_tests]) .- abs.(lm_coefs[n_tests+1:end])
scatter(knockoff_mirror_stat)

opt_t = MirrorStatistic.get_t(knockoff_mirror_stat; fdr_target=fdr_level)

MirrorStatistic.false_discovery_rate(
    true_coef=beta_true .!= 0,
    estimated_coef=knockoff_mirror_stat .> opt_t
)
MirrorStatistic.wrapper_metrics(
    beta_true .!= 0,
    knockoff_mirror_stat .> opt_t
)

# Check expectation
Random.seed!(134)
n_rep = 100
fdr_with_correction = []
lm_coef_dist = []

for rep = 1:n_rep

    Xr = rand(dist_population, N)'
    yr = Xr * beta_true .+ randn(N)

    X_k = rand(dist_population, N)'
    X_a = hcat(Xr, X_k)

    # simple lm
    lm_result = GLMNet.glmnetcv(X_a, yr)
    lm_coefs = GLMNet.coef(lm_result)
    push!(lm_coef_dist, lm_coefs)

    knockoff_mirror_stat = abs.(lm_coefs[1:n_tests]) .- abs.(lm_coefs[n_tests+1:end])

    opt_t = MirrorStatistic.get_t(knockoff_mirror_stat; fdr_target=fdr_level)

    push!(
        fdr_with_correction,
        MirrorStatistic.false_discovery_rate(
            true_coef=beta_true .!= 0,
            estimated_coef=knockoff_mirror_stat .> opt_t
        )
    )
end

mean(fdr_with_correction)
histogram(fdr_with_correction)

histogram(hcat(lm_coef_dist...)[n_tests, :])

# ----------------------------------------------------
# Bayesian Gaussian model
function normal_posterior(x_mean, mu_0, sigma2_0, sigma2, n)
    mu_p = x_mean * sigma2_0 / (sigma2 / n + sigma2_0) + mu_0 * sigma2 / (sigma2 / n + sigma2_0)
    sigma2_p = 1 / (1 / sigma2_0 + n / sigma2)
    dist_p = Normal(mu_p, sqrt(sigma2_p))

    return dist_p
end

function normal_prior(mu_0, sigma2_0)
    dist_p = Normal(mu_0, sqrt(sigma2_0))
    return dist_p
end

function prod_prior(beta_mean, tau)
    if typeof(tau) <: Distributions.Distribution
        tau_s = rand(tau)
    else
        tau_s = tau
    end
    
    lambda = beta_mean

    eta = Normal(0., lambda * tau_s)
    dist_p = eta * lambda
    return dist_p
end

normal_prior(0., 1.)
prod_prior(0.5, 1)

# E(FDR) integrating over the prior distribution
density(rand(prod_prior(0.5, 1), 500))
density!(rand(normal_prior(0., 1.), 500))

# Posterior - assume the posterior distributions are Gaussian
n_fn = 0
n_fp = 2
posterior_mu = vcat(
    ones(n_true - n_fn),
    zeros(n_fn),
    zeros(n_null - n_fp),
    ones(n_fp) * 0.3
)

posterior_sigma = vcat(
    ones(n_true) * 0.3,
    ones(n_null - n_fp) * 0.3,
    ones(n_fp) * 0.5
)
posterior_dist = Distributions.Normal.(posterior_mu, posterior_sigma)

samples_posterior = hcat(rand.(posterior_dist, 300)...)
density(samples_posterior, label=false)


mean_vec = MirrorStatistic.mean_folded_normal.(posterior_mu, posterior_sigma) .- 
    MirrorStatistic.mean_folded_normal.(0., posterior_sigma)
var_vec = MirrorStatistic.var_folded_normal.(posterior_mu, posterior_sigma) .+ 
    MirrorStatistic.var_folded_normal.(0., posterior_sigma)

ms_dist_vec = [
    Distributions.Normal(mean_ms, sqrt(var_ms)) for (mean_ms, var_ms) in zip(mean_vec, var_vec)
]

mirror_coeff = rand.(ms_dist_vec)
scatter(mirror_coeff)

opt_t = MirrorStatistic.get_t(mirror_coeff; fdr_target=fdr_level)
sum(mirror_coeff .> opt_t)
sum(mirror_coeff .< -opt_t)

sum(mirror_coeff .< -opt_t) / sum(mirror_coeff .> opt_t)

MirrorStatistic.false_discovery_rate(
    true_coef=beta_true .!= 0,
    estimated_coef=mirror_coeff .> opt_t
)

M = 150
mirror_coeff = hcat(rand.(ms_dist_vec, M)...)'
scatter(mirror_coeff, label=false)

opt_t = MirrorStatistic.get_t(mirror_coeff; fdr_target=fdr_level)
sum(mirror_coeff .> opt_t) / M
sum(mirror_coeff .< -opt_t) / M

sum(mirror_coeff .< -opt_t) / sum(mirror_coeff .> opt_t)

t, selection = MirrorStatistic.posterior_fdr_threshold(vec(mean(mirror_coeff .> opt_t, dims=2)), fdr_level)
MirrorStatistic.wrapper_metrics(
    beta_true .!= 0,
    selection
)


# B-FDR, Monte Carlo approximation
Random.seed!(134)
mc_samples = 1000

fdp = []
fdp_estimated = []
fp_estimated = []
sum_included = []
matrix_mirror_coeff = zeros(n_tests, mc_samples)
thresholds = []
matrix_included = zeros(n_tests, mc_samples)
matrix_below_t = zeros(n_tests, mc_samples)


for mc = 1:mc_samples

    mirror_coeff = MirrorStatistic.mirror_statistic(rand.(posterior_dist), rand.(posterior_dist))

    # mirror_coeff = rand.(ms_dist_vec)
    matrix_mirror_coeff[:, mc] = mirror_coeff

    opt_t = MirrorStatistic.get_t(mirror_coeff; fdr_target=fdr_level)
    push!(thresholds, opt_t)
    push!(sum_included, sum(mirror_coeff .> opt_t))
    push!(fp_estimated, sum(mirror_coeff .< -opt_t))

    push!(fdp_estimated, sum(mirror_coeff .< -opt_t) / sum(mirror_coeff .> opt_t))
    matrix_included[:, mc] = mirror_coeff .> opt_t
    matrix_below_t[:, mc] = mirror_coeff .< opt_t

    push!(fdp,
        MirrorStatistic.false_discovery_rate(
            true_coef=beta_true .!= 0,
            estimated_coef=mirror_coeff .> opt_t
        )
    )
end

mean(fdp)
histogram(fdp)

mean(fdp_estimated)
histogram(fdp_estimated)

mean(fp_estimated)
mean(sum_included)
histogram(fp_estimated)
histogram(sum_included)
histogram(thresholds)

mean(sum_included)
median(sum_included)

scatter(matrix_mirror_coeff[:, 1])


inclusion_probs = mean(matrix_included, dims=2)[:, 1]
sorted_indeces = sortperm(inclusion_probs, rev=true)
sorted_indeces_rev = sortperm(inclusion_probs, rev=false)
scatter(inclusion_probs[sorted_indeces])

t, selection = MirrorStatistic.posterior_fdr_threshold(inclusion_probs, 0.1)
MirrorStatistic.wrapper_metrics(
    beta_true .!= 0,
    selection
)

# One global threshold
fdr_level = 0.1
optimal_t = MirrorStatistic.get_t(matrix_mirror_coeff; fdr_target=fdr_level)

L = vec(sum(matrix_mirror_coeff .< -optimal_t, dims=2))
R = vec(sum(matrix_mirror_coeff .> optimal_t, dims=2))
log_ratio = log.(L .+ 1) .- log.(R .+ 1)

log_ratio = log.(L .+ 1) .- log.(R)
selection = log_ratio .<= log((fdr_level*n_tests + 1) / n_tests)

ratio = L ./ R
selection = ratio .<= fdr_level
scatter(ratio)
hline!([0.1])

MirrorStatistic.wrapper_metrics(
    beta_true .!= 0,
    selection
)


scatter(R / mc_samples)
hline!([0.1])
selection = R / mc_samples .>= fdr_target

MirrorStatistic.wrapper_metrics(
    beta_true .!= 0,
    selection
)


# ------- relative inclusion frequency -------
selection_dimension = sum(matrix_included, dims=1)

relative_freq = mean(matrix_included ./ selection_dimension, dims=2)[:, 1]
sorted_indeces = sortperm(relative_freq, rev=false)
scatter(relative_freq[sorted_indeces], ms=2)

relative_freq = mean(matrix_included ./ maximum(selection_dimension), dims=2)[:, 1]
sorted_indeces = sortperm(relative_freq, rev=false)
scatter!(relative_freq[sortperm(relative_freq, rev=false)], ms=2)

relative_freq = mean(matrix_included ./ mean(selection_dimension), dims=2)[:, 1]
sorted_indeces = sortperm(relative_freq, rev=false)
scatter!(relative_freq[sortperm(relative_freq, rev=false)], ms=2)

cumsum_rel_freq = cumsum(relative_freq[sorted_indeces])
scatter(cumsum_rel_freq)
hline!([0.1])
max_t = sum(cumsum_rel_freq .<= 0.1)

cumsum_rel_freq .> cumsum_rel_freq[max_t]
selection = zeros(n_tests)
selection[sorted_indeces[max_t:end]] .= 1
sum(selection)

MirrorStatistic.wrapper_metrics(
    beta_true .!= 0,
    selection
)

selection = MirrorStatistic.relative_inclusion_frequency(matrix_included, 0.1, "max")
MirrorStatistic.wrapper_metrics(
    beta_true .!= 0,
    selection
)


# difference in inclusion probabilities
inclusion_probs = mean(matrix_included, dims=2)[:, 1]
selection_dimension = sum(matrix_included, dims=1)
relative_freq = mean(matrix_included ./ selection_dimension, dims=2)[:, 1]
max_relative_freq = mean(matrix_included ./ maximum(selection_dimension), dims=2)[:, 1]
mean_relative_freq = mean(matrix_included ./ mean(selection_dimension), dims=2)[:, 1]

scatter(inclusion_probs, label="probs")
scatter(relative_freq, label="rel")
scatter!(max_relative_freq, label="max")
scatter!(mean_relative_freq, label="mean")


# Real FDR - expectation over repeated sampling
Random.seed!(134)
n_rep = 100
fdr_with_correction = []
fdr_no_correction = []

for rep = 1:n_rep

    Xr = rand(dist_population, N)

    sample_mean = mean(Xr, dims=2)

    dist_post = normal_posterior.(sample_mean, 0., 100., 1., N)

    q0 = quantile.(dist_post, 0.025)
    q1 = quantile.(dist_post, 0.975)
    push!(
        fdr_no_correction,
        MirrorStatistic.false_discovery_rate(
            true_coef=mu_population .!= 0,
            estimated_coef=(q0 .* q1) .> 0
        )
    )

    # MS
    mirror_coeff = MirrorStatistic.mirror_statistic(
        vcat(rand.(dist_post, 100)...),
        vcat(rand.(dist_post, 100)...)
    )
    opt_t = MirrorStatistic.get_t(mirror_coeff; fdr_target=fdr_level)

    push!(
        fdr_with_correction,
        MirrorStatistic.false_discovery_rate(
            true_coef=mu_population .!= 0,
            estimated_coef=mirror_coeff .> opt_t
        )
    )
end

mean(fdr_with_correction)
histogram(fdr_with_correction)

mean(fdr_no_correction)
histogram(fdr_no_correction)

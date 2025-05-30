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

abs_project_path = normpath(joinpath(@__FILE__, ".."))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))

results_path = joinpath(abs_project_path, "results", "ms_analysis")

# Frequentist FDR is defined as E(FDP)
# FDP = False Discovery Proportion

n_tests = 30
n_true = 10
n_null = n_tests - n_true
alpha = 0.05
fdr_level = 0.1
# sample size
N = 100


# Generate N samples from the n-dimensional population
mu_population = zeros(n_tests)
dist_population = MultivariateNormal(ones(n_tests))
beta_true = vcat(
    [1., 1., -1., -1., 1., 1., 1., -1., -1., 1.],
    zeros(n_null)
)

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

lm_result = lm(X_a, y)
lm_coefs = coeftable(lm_result).cols[1]

knockoff_mirror_stat = abs.(lm_coefs[1:n_tests]) .- abs.(lm_coefs[n_tests+1:end])
scatter(knockoff_mirror_stat)

opt_t = MirrorStatistic.get_t(knockoff_mirror_stat; fdr_target=fdr_level)

MirrorStatistic.false_discovery_rate(
    true_coef=beta_true .!= 0,
    estimated_coef=knockoff_mirror_stat .> opt_t
)

# Check expectation
Random.seed!(134)
n_rep = 100
fdr_with_correction = []

for rep = 1:n_rep

    Xr = rand(dist_population, N)'
    yr = Xr * beta_true .+ randn(N)

    X_k = rand(dist_population, N)'
    X_a = hcat(Xr, X_k)

    # simple lm
    lm_result = lm(X_a, yr)
    lm_coefs = coeftable(lm_result).cols[1]

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

# Product prior
# Posterior - assume the posterior distributions are Gaussian
posterior_mu = beta_true
posterior_sigma = ones(n_tests) * 0.2

MirrorStatistic.mean_folded_normal.(posterior_mu, posterior_sigma)
MirrorStatistic.var_folded_normal.(posterior_mu, posterior_sigma)

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


# Local-FDR, Monte Carlo approximation
Random.seed!(134)
mc_samples = 1000

fdp = []
fdp_estimated = []
fp_estimated = []
sum_included = []

for mc = 1:mc_samples
    mirror_coeff = rand.(ms_dist_vec)

    opt_t = MirrorStatistic.get_t(mirror_coeff; fdr_target=fdr_level)
    push!(sum_included, sum(mirror_coeff .> opt_t))
    push!(fp_estimated, sum(mirror_coeff .< -opt_t))

    push!(fdp_estimated, sum(mirror_coeff .< -opt_t) / sum(mirror_coeff .> opt_t))

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


# Expected value
p = 10
dist = prod_prior(0.5, 1)



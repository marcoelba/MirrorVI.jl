# Gaussian Mirrors
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


p = 100
p0 = 90
p1 = 10
fdr_target = 0.1
beta = cat(zeros(p0), ones(p1), dims=1)
dist_population = MultivariateNormal(ones(p))

n = 1000

function generate_data(n, beta)
    X = rand(dist_population, n)'
    y = X * beta .+ randn(n) * 0.5
    return X, y
end 

ms_abs(half_1, half_2) = @. abs(half_1 + half_2) - abs(half_1 - half_2)
ms_square(half_1, half_2) = @. (half_1 + half_2)^2 - (half_1 - half_2)^2

function take_halves(x)
    lx = Int(size(x, 1) / 2)
    return x[1:lx, :], x[lx+1:size(x, 1), :]
end

# Check
X, y = generate_data(n, beta)
# simple parametric regression
lm_result = lm(X, y)
lm_pvalues = coeftable(lm_result).cols[4]
sum(lm_pvalues .< fdr_target)

# Uniform distribution of p-values under the null hypothesis
MirrorStatistic.false_discovery_rate(
    true_coef=beta .!= 0,
    estimated_coef=lm_pvalues .< fdr_target
)

# BH correction for FDR
bh_pvalues = MirrorStatistic.bh_correction(p_values=lm_pvalues, fdr_level=fdr_target)
MirrorStatistic.false_discovery_rate(
    true_coef=beta .!= 0,
    estimated_coef=bh_pvalues[:, 3]
)

# --------------------------------------------------------------
# FDR is defined as the expectation over repeated sampling
# so, do repeated sampling
Random.seed!(134)
n_rep = 500
fdr_no_correction = []
fdr_with_correction = []
lm_coeffs = zeros(p, n_rep)

for rep = 1:n_rep

    X, y = generate_data(n, beta)
    # simple lm
    lm_result = lm(X, y)
    lm_pvalues = coeftable(lm_result).cols[4]
    lm_coeffs[:, rep] = coeftable(lm_result).cols[1]
    
    push!(
        fdr_no_correction,
        MirrorStatistic.false_discovery_rate(
            true_coef=beta .!= 0,
            estimated_coef=lm_pvalues .< fdr_target
        )
    )
    # BH correction for FDR
    bh_pvalues = MirrorStatistic.bh_correction(p_values=lm_pvalues, fdr_level=fdr_target)
    push!(
        fdr_with_correction,
        MirrorStatistic.false_discovery_rate(
            true_coef=beta .!= 0,
            estimated_coef=bh_pvalues[:, 3]
        )
    )
end

mean(fdr_no_correction)
mean(fdr_with_correction)
histogram(fdr_no_correction)
histogram(fdr_with_correction)

histogram(lm_coeffs[1, :])
histogram(lm_coeffs[p, :])

# Mirror Statistic
# taking random draws from the sampling distributin of beta
emp_beta_0 = lm_coeffs[p, :]
half_1 = emp_beta_0[1:Int(n_rep/2)]
half_2 = emp_beta_0[Int(n_rep/2 + 1):n_rep]

mirror_stat = zeros(Int(n_rep/2), p)
optimal_t = zeros(Int(n_rep/2))
for jj = 1:p
    emp_beta = lm_coeffs[jj, :]
    half_1 = emp_beta[1:Int(n_rep/2)]
    half_2 = emp_beta[Int(n_rep/2 + 1):n_rep]
    ms = ms_abs(half_1, half_2)
    # ms_square(half_1, half_2)
    mirror_stat[:, jj] = ms
    opt_t = MirrorStatistic.get_t(mirror_stat; fdr_target=fdr_target)
end
histogram(mirror_stat, label=false)
histogram(mirror_stat[1, :], label=false)
opt_t = MirrorStatistic.get_t(mirror_stat; fdr_target=fdr_target)
mirror_stat .> opt_t

# one variable
mirror_stat[:, 1]
mean(mirror_stat[:, 1] .> 0)
mean(mirror_stat[:, p] .> 0)


# --------------------------------------------------------------
# Bayesian Gaussian Mirrors
n_samples = 10000

# Prior Distribution
function prior_distribution(;lambda=nothing, a=nothing, b=nothing)
    if isnothing(lambda)
        lambda = rand(Beta(a, b))
    end
    prior = Normal(0, lambda^2)
    return prior
end

function savs_calibration(posterior_sample, x)
    mu = mean(posterior_sample)
    if (abs(mu) * norm(x)) < (1 / abs(mu)^2)
        mu_star = 0
    else
        mu_star = mu
    end
    return mu_star
end

function newton_calibration(ms_mean_inclusion, fdr_target)
    fp_prob = 1. .- ms_mean_inclusion
    c_opt = 0.

    for c in sort(fp_prob, rev=true, dims=1)
        lower_than_c = fp_prob .<= c
        if (sum(fp_prob[lower_than_c]) / sum(lower_than_c)) < fdr_target
            c_opt = c
            break
        end
    end
    return fp_prob .<= c_opt
end

X, y = generate_data(n, beta)

sample_from_prior = rand(prior_distribution(lambda=0.01), n_samples)
histogram(sample_from_prior)

sample_from_prior = rand(prior_distribution(a=0.001, b=1.), n_samples)
histogram(sample_from_prior)

# Posterior Distribution
function posterior_distribution(; mu, lambda)
    dist = Normal.(mu, lambda^2)
    return dist
end

posterior_null = posterior_distribution(mu=zeros(p0) .+ 0.0, lambda=0.2)
posterior_alternative = posterior_distribution(mu=ones(p1), lambda=0.2)
posterior = cat(posterior_null, posterior_alternative, dims=1)

Random.seed!(1243)
sample_from_posterior = rand.(posterior, n_samples)
sample_from_posterior = hcat(sample_from_posterior...)

histogram(sample_from_posterior, label=false)
mean(sample_from_posterior[:, 3] .> 0.)

# MS
ms = ms_abs(take_halves(sample_from_posterior)...)
histogram(ms, label=false)

opt_t = MirrorStatistic.get_t(ms; fdr_target=fdr_target)
MirrorStatistic.false_discovery_rate(
    true_coef=beta .!= 0,
    estimated_coef=mean(ms, dims=1)' .> opt_t
)

# Explicit Prob approach
function fdp_from_ms_prob(ms, t)
    tot_prob_fp = sum([mean(ms[:, jj] .< -t) for jj = 1:p])
    tot_prob_selection = sum([mean(ms[:, jj] .> t) for jj = 1:p])
    fdp = tot_prob_fp / (tot_prob_selection + 1 * (tot_prob_selection == 0))
    return fdp, tot_prob_fp, tot_prob_selection
end

fdp_dist = []
t_steps = []
fp_dist = []
tot_dist = []
for t in range(0, maximum(ms), length=1500)
    push!(t_steps, t)
    fdp_t, tot_prob_fp, tot_prob_selection = fdp_from_ms_prob(ms, t)
    push!(fdp_dist, fdp_t)
    push!(fp_dist, tot_prob_fp)
    push!(tot_dist, tot_prob_selection)
end
t_opt = sum(1 .- (fdp_dist .< fdr_target))
t_steps[t_opt]
fdp_dist[t_opt]
fp_dist[t_opt]
tot_dist[t_opt]
inclusion_prob = [mean(ms[:, jj] .> t_steps[t_opt]) for jj = 1:p]

sort_inclusion_prob = sort(inclusion_prob, rev=true)
sort_fp_prob = 1 .- sort_inclusion_prob
for jj = 1:p
    fdp_jj = sum(sort_fp_prob[1:jj]) / jj
    if fdp_jj >= 0.1
        println(fdp_jj)
        println(jj)
        break
    end
end


scatter(t_steps, fdp_dist .* 100, label="FDP", markersize=2)
scatter!(t_steps, fp_dist, label="FP", markersize=2)
scatter!(t_steps, tot_dist, label="TOT", markersize=2)
vline!([t_steps[t_opt]], color="red", linewidth=1)
hline!([fdr_target*100], color="red", linewidth=1)
ylabel!("FDP")
xlabel!("t")


# With SAVS
mu_star = []
for jj = 1:p
    push!(mu_star, savs_calibration(sample_from_posterior[:, jj], X[:, jj]))
end
# calibrate the means of the posterior
for jj = 1:p
    if mu_star[jj] == 0
        sample_from_posterior[:, jj] .-= posterior[jj].μ
    end
end

ms = ms_abs(take_halves(sample_from_posterior)...)
opt_t = MirrorStatistic.get_t(ms; fdr_target=fdr_target)
ms_mean_inclusion = mean(ms, dims=1)' .> opt_t

histogram(ms, label=false, color="lightgrey", alpha=0.8)
vline!([opt_t], color="red", linewidth=2)

MirrorStatistic.false_discovery_rate(
    true_coef=beta .!= 0,
    estimated_coef=ms_mean_inclusion
)

# Newton
newton_selection = newton_calibration(mean(ms .> opt_t, dims=1)', fdr_target)
MirrorStatistic.false_discovery_rate(
    true_coef=beta .!= 0,
    estimated_coef=newton_selection
)


# One sample at a time
Random.seed!(1243)
sample_from_posterior = rand.(posterior, n_samples)
sample_from_posterior = hcat(sample_from_posterior...)

# MS
ms = ms_abs(take_halves(sample_from_posterior)...)
ms = ms_square(take_halves(sample_from_posterior)...)

opt_t = []
for ms_sample in eachrow(ms)
    push!(opt_t, MirrorStatistic.get_t(ms_sample; fdr_target=fdr_target))
end
ms_mean_inclusion = mean(ms .> opt_t; dims=1)'
ms_mean_selection = mean(ms, dims=1)' .> mean(opt_t)
MirrorStatistic.false_discovery_rate(
    true_coef=beta .!= 0,
    estimated_coef=ms_mean_selection
)

# With SAVS
mu_star = []
for jj = 1:p
    push!(mu_star, savs_calibration(sample_from_posterior[:, jj], X[:, jj]))
end
# calibrate the means of the posterior
for jj = 1:p
    if mu_star[jj] == 0
        sample_from_posterior[:, jj] .-= posterior[jj].μ
    end
end

ms = ms_abs(take_halves(sample_from_posterior)...)
ms = ms_square(take_halves(sample_from_posterior)...)

opt_t = MirrorStatistic.get_t(ms; fdr_target=fdr_target)
ms_mean_inclusion = mean(ms .> opt_t, dims=1)'

histogram(ms, label=false, color="lightgrey", alpha=0.8)
vline!([opt_t], color="red", linewidth=2)

MirrorStatistic.false_discovery_rate(
    true_coef=beta .!= 0,
    estimated_coef=mean(ms, dims=1)' .> opt_t
)

# Newton
newton_selection = newton_calibration(ms_mean_inclusion, fdr_target)
MirrorStatistic.false_discovery_rate(
    true_coef=beta .!= 0,
    estimated_coef=newton_selection
)

histogram(ms[:, 1:p0], label=false)

# Variance-Gamma distribution
dist = Normal(0, 0.9)
x = rand(dist, 1000)
z_dist(dist) = 4 * rand(dist, 1000) .* rand(dist, 1000)
z = z_dist(dist)

histogram(x)
histogram!(z, alpha=0.5)
var(x)
var(z)
(4 * 0.9^2)^2

# more Zs
k = 1000
z_sum = sum([z_dist(dist) for ii = 1:k])
var(z_sum)
k*(4 * 0.9^2)^2

histogram(z_sum)
histogram!(rand(Normal(0, sqrt(k * (4 * 0.9^2)^2)), 1000))

# Simulations linear model
using MirrorVI
using MirrorVI: update_parameters_dict, DistributionsLogPdf, VariationalDistributions, LogExpFunctions, Predictors, mean

using CSV
using DataFrames
using StatsPlots
using GLM

abs_project_path = normpath(joinpath(@__FILE__, ".."))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))


n_individuals = 300

p = 1000
prop_non_zero = 0.025
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5

num_iter = 1000
MC_SAMPLES = 5000
fdr_target = 0.1
n_simulations = 30
random_seed = 1234
simulations_models = Dict()
simulations_metrics = Dict()

# label_files = "algo_HS_linear_n$(n_individuals)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"
label_files = "algo_product_linear_n$(n_individuals)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"

params_dict = MirrorVI.OrderedDict()

# beta 0
update_parameters_dict(
    params_dict;
    name="beta0",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_normal(x),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=identity),
    init_z=[0., 0.1]
)

# beta fixed
update_parameters_dict(
    params_dict;
    name="tau_beta",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_half_cauchy(x, sigma=1.),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=vcat(0.1, 0.1)
)

update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dim_theta=(p, ),
    logpdf_prior=x::AbstractArray -> DistributionsLogPdf.log_beta(x, 1, 1),
    dim_z=p*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.logistic),
    init_z=vcat(randn(p)*0.1, randn(p)*0.1),
    dependency=[]
)
update_parameters_dict(
    params_dict;
    name="beta",
    dim_theta=(p, ),
    logpdf_prior=(x::AbstractArray, sigma::AbstractArray, tau::Real) -> DistributionsLogPdf.log_normal(x, sigma=sigma .* tau),
    dim_z=p*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
    init_z=vcat(randn(p)*0.01, randn(p)*0.01),
    dependency=["sigma_beta", "tau_beta"]
)

# sigma y
update_parameters_dict(
    params_dict;
    name="sigma_y",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_half_normal(x, 1.),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=vcat(1., 0.5)
)

prior_position = params_dict["tuple_prior_position"]

function linear_model(
    theta::MirrorVI.ComponentArray;
    X::AbstractArray,
    )
    n = size(X, 1)

    mu = theta[:beta0] .+ X * (theta[:beta] .* theta[:sigma_beta])
    sigma = ones(n) .* theta[:sigma_y]

    return (mu, sigma)
end


for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = MirrorVI.generate_linear_model_data(
        n_individuals=n_individuals,
        p=p, p1=p1, p0=p0, corr_factor=corr_factor,
        random_seed=random_seed + simu
    )
    y = data_dict["y"]
    X = data_dict["X"]
    y = y .- mean(y)
        
    # Training
    z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
    optimiser = MirrorVI.MyOptimisers.DecayedADAGrad()
    
    res = MirrorVI.hybrid_training_loop(
        z=z,
        y=y,
        X=X,
        params_dict=params_dict,
        model=linear_model,
        log_likelihood=DistributionsLogPdf.log_normal,
        log_prior=x::AbstractArray -> MirrorVI.compute_logpdf_prior(x; params_dict=params_dict),
        n_iter=num_iter,
        optimiser=optimiser,
        save_all=false,
        use_noisy_grads=false,
        elbo_samples=3
    )
    
    z = res["best_iter_dict"]["best_z"]

    q = VariationalDistributions.get_variational_dist(
        z,
        params_dict["vi_family_array"],
        params_dict["ranges_z"]
    )
    
    simulations_models[simu] = (q, res["loss_dict"]["loss"], params_dict)

    # ------ Mirror Statistic ------
    metrics_dict = Dict()

    # ms_dist = MirrorStatistic.posterior_ms_coefficients(q[prior_position[:beta]].dist)

    # metrics = MirrorStatistic.optimal_inclusion(
    #     ms_dist_vec=ms_dist,
    #     mc_samples=MC_SAMPLES,
    #     beta_true=data_dict["beta"],
    #     fdr_target=0.1
    # )
    # metrics_dict = Dict()

    # # Posterior
    # inclusion_probs = mean(metrics.inclusion_matrix, dims=2)[:, 1]
    # c_opt, selection = MirrorStatistic.posterior_fdr_threshold(inclusion_probs, fdr_target)

    # metrics_posterior = MirrorStatistic.wrapper_metrics(
    #     data_dict["beta"] .!= 0.,
    #     selection
    # )
    
    # metrics_dict["metrics_posterior"] = metrics_posterior

    # Monte Carlo loop
    mc_samples = 5000
    ms_samples = Int(mc_samples / 2)
    # mean_sigma = mean(rand(q[prior_position[:sigma_beta]], mc_samples), dims=2)[:, 1]
    beta = rand(q[prior_position[:beta]], mc_samples)

    ms_coeffs = MirrorStatistic.mirror_statistic(beta[:, 1:ms_samples], beta[:, ms_samples+1:mc_samples])
    opt_t = MirrorStatistic.get_t(ms_coeffs; fdr_target=fdr_target)
    inclusion_matrix = ms_coeffs .> opt_t
    mean_inclusion_per_coef = mean(inclusion_matrix, dims=2)[:, 1]

    c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
        mean_inclusion_per_coef,
        fdr_target
    )

    metrics_mc = MirrorStatistic.wrapper_metrics(
        data_dict["beta"] .!= 0.,
        selection
    )
    metrics_dict["metrics_mc"] = metrics_mc

    # Using the distributions
    R = vec(sum(ms_coeffs .> opt_t, dims=2))
    selection = R / ms_samples .>= fdr_target
    metrics_R = MirrorStatistic.wrapper_metrics(
        data_dict["beta"] .!= 0.,
        selection
    )

    metrics_dict["metrics_R"] = metrics_R


    simulations_metrics[simu] = metrics_dict

end


mc_fdr = []
mc_tpr = []
posterior_fdr = []
posterior_tpr = []

for simu = 1:n_simulations
    push!(mc_fdr, simulations_metrics[simu]["metrics_mc"].fdr)
    push!(mc_tpr, simulations_metrics[simu]["metrics_mc"].tpr)

    push!(posterior_fdr, simulations_metrics[simu]["metrics_R"].fdr)
    push!(posterior_tpr, simulations_metrics[simu]["metrics_R"].tpr)
end

all_metrics = hcat(mc_fdr, posterior_fdr, mc_tpr, posterior_tpr)
df = DataFrame(all_metrics, ["mc_fdr", "posterior_fdr", "mc_tpr", "posterior_tpr"])

CSV.write(
    joinpath(abs_project_path, "results", "simulations", "$(label_files).csv"),
    df
)


plt = violin([1], posterior_fdr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([1], posterior_fdr, label=false, color="blue", fillalpha=0.1, linewidth=2)

violin!([2], posterior_tpr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([2], posterior_tpr, label=false, color="blue", fillalpha=0.1, linewidth=2)

xticks!([1, 2], ["FDR", "TPR"], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot.pdf"))

# MC
plt = violin([1], mc_fdr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([1], mc_fdr, label=false, color="blue", fillalpha=0.1, linewidth=2)

violin!([2], mc_tpr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([2], mc_tpr, label=false, color="blue", fillalpha=0.1, linewidth=2)

xticks!([1, 2], ["FDR", "TPR"], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot.pdf"))

# fdr
plt = violin([1], mc_fdr, color="lightblue", label="MC", alpha=1, linewidth=0)
violin!([2], posterior_fdr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([1], mc_fdr, label=false, color="blue", fillalpha=0., linewidth=2)
boxplot!([2], posterior_fdr, label=false, color="blue", fillalpha=0., linewidth=2)

# tpr
violin!([4], mc_tpr, color="lightblue", label=false, alpha=1, linewidth=0)
violin!([5], posterior_tpr, color="lightblue", label=false, alpha=1, linewidth=0)

boxplot!([4], mc_tpr, label=false, color="blue", fillalpha=0., linewidth=2)
boxplot!([5], posterior_tpr, label=false, color="blue", fillalpha=0., linewidth=2)

xticks!([1, 2, 4, 5], ["FDR", "TPR"], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)


# ----------- Data Splitting -----------
ds_fdr = []
ds_tpr = []

for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = MirrorVI.generate_linear_model_data(
        n_individuals=n_individuals,
        p=p, p1=p1, p0=p0, corr_factor=corr_factor,
        random_seed=random_seed + simu
    )

    X1=Float64.(data_dict["X"][1:Int(n_individuals/2), :])
    X2=Float64.(data_dict["X"][Int(n_individuals/2)+1:end, :])
    y1=Float64.(data_dict["y"][1:Int(n_individuals/2)])
    y2=Float64.(data_dict["y"][Int(n_individuals/2)+1:end])

    lasso = GLMNet.glmnetcv(X1, y1)
    lasso_coef = GLMNet.coef(lasso)
    # Non-0 coefficients
    non_zero = lasso_coef .!= 0

    X2 = X2[:, non_zero]
    p1 = size(X2)[2]
    X2 = hcat(X2, ones(size(X2)[1], 1))
    p1 += 1
    lm_on_split2 = GLM.lm(X2, y2)

    lm_on_v_coef = GLM.coef(lm_on_split2)[1:(p1 - 1)]
    lm_coef = zeros(length(lasso_coef))
    lm_coef[non_zero] = lm_on_v_coef

    beta_1 = lasso_coef
    beta_2 = lm_coef
    
    mirror_coeffs = MirrorStatistic.mirror_statistic(beta_1, beta_2)

    opt_t = MirrorStatistic.get_t(mirror_coeffs; fdr_target=fdr_target)

    metrics = MirrorStatistic.wrapper_metrics(
        data_dict["beta"] .!= 0.,
        mirror_coeffs .> opt_t
    )

    push!(ds_fdr, metrics.fdr)
    push!(ds_tpr, metrics.tpr)

end

plt = violin([1], ds_fdr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([1], ds_fdr, label=false, color="blue", fillalpha=0.1, linewidth=2)

violin!([2], ds_tpr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([2], ds_tpr, label=false, color="blue", fillalpha=0.1, linewidth=2)

xticks!([1, 2], ["FDR", "TPR"], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot_DS.pdf"))


# --------------------------------------------------------------------
# Single Run of Bayesian Model

data_dict = MirrorVI.generate_linear_model_data(
    n_individuals=n_individuals,
    p=p, p1=p1, p0=p0, corr_factor=corr_factor,
    random_seed=random_seed
)
y = data_dict["y"]
X = data_dict["X"]
y = y .- mean(y)


params_dict = MirrorVI.OrderedDict()

# beta 0
update_parameters_dict(
    params_dict;
    name="beta0",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_normal(x),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=identity),
    init_z=[0., 0.1]
)

# beta fixed
update_parameters_dict(
    params_dict;
    name="tau_beta",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_half_cauchy(x, sigma=1.),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=vcat(0.1, 0.1)
)

update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dim_theta=(p, ),
    logpdf_prior=x::AbstractArray -> DistributionsLogPdf.log_beta(x, 1, 1),
    dim_z=p*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.logistic),
    init_z=vcat(randn(p)*0.1, randn(p)*0.1),
    dependency=[]
)
update_parameters_dict(
    params_dict;
    name="beta",
    dim_theta=(p, ),
    # logpdf_prior=(x::AbstractArray, sigma::AbstractArray, tau::Real) -> sum(logpdf_mixture(
    #     x, sigma, tau, 100.
    #     )),
    logpdf_prior=(x::AbstractArray, sigma::AbstractArray, tau::Real) -> DistributionsLogPdf.log_normal(
        x, sigma=sigma .* tau
    ),
    dim_z=p*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
    init_z=vcat(randn(p)*0.01, randn(p)*0.01),
    dependency=["sigma_beta", "tau_beta"]
)

# sigma y
update_parameters_dict(
    params_dict;
    name="sigma_y",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_half_normal(x, 1.),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=vcat(1., 0.5)
)

prior_position = params_dict["tuple_prior_position"]

function logpdf_mixture(x::Real, tau::Real, p::Real, c::Real)
    # Compute the log-pdf of the first normal component (variance = 10 * tau)
    log_pdf1 = -0.5 * (log(2π * c * tau) + (x^2) / (c * tau))
    
    # Compute the log-pdf of the second normal component (variance = tau)
    log_pdf2 = -0.5 * (log(2π * tau) + (x^2) / tau)
    
    # Combine them using log-sum-exp for numerical stability
    log_mixture = log(p * exp(log_pdf1) + (1 - p) * exp(log_pdf2))
    
    return log_mixture
end
logpdf_mixture(1., 0.1, 0.3, 100.)
logpdf_mixture(0.5, 0.1, 0.6, 100.)


function logpdf_mixture(x::AbstractArray, p::AbstractArray, tau::Real, c::Real)
    # Compute the log-pdf of the first normal component (variance = 10 * tau)
    log_pdf1 = -0.5 .* (log(2π * c * tau) .+ (x.^2) ./ (c * tau))
    
    # Compute the log-pdf of the second normal component (variance = tau)
    log_pdf2 = -0.5 .* (log(2π * tau) .+ (x.^2) ./ tau)
    
    # Combine them using log-sum-exp for numerical stability
    log_mixture = log.(p .* exp.(log_pdf1) .+ (1 .- p) .* exp.(log_pdf2))
    
    return log_mixture
end
logpdf_mixture([1., 0.5], [0.3, 0.6], 0.1, 100.)



function linear_model(
    theta::MirrorVI.ComponentArray;
    X::AbstractArray,
    )
    n = size(X, 1)

    mu = theta[:beta0] .+ X * (theta[:beta] .* theta[:sigma_beta])
    sigma = ones(n) .* theta[:sigma_y]

    return (mu, sigma)
end

# Training
z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
optimiser = MirrorVI.MyOptimisers.DecayedADAGrad()

res = MirrorVI.hybrid_training_loop(
    z=z,
    y=y,
    X=X,
    params_dict=params_dict,
    model=linear_model,
    log_likelihood=DistributionsLogPdf.log_normal,
    log_prior=x::AbstractArray -> MirrorVI.compute_logpdf_prior(x; params_dict=params_dict),
    n_iter=num_iter,
    optimiser=optimiser,
    save_all=false,
    use_noisy_grads=false,
    elbo_samples=3
)

z = res["best_iter_dict"]["best_z"]

q = VariationalDistributions.get_variational_dist(
    z,
    params_dict["vi_family_array"],
    params_dict["ranges_z"]
)

plot(res["loss_dict"]["loss"])
beta_samples = rand(q[prior_position[:beta]], 500)
sigma_beta_samples = rand(q[prior_position[:sigma_beta]], 500)
tau_samples = rand(q[prior_position[:tau_beta]], 500)

density(beta_samples', label=false)
density(beta_samples' .* sigma_beta_samples' .* tau_samples, label=false)
density(tau_samples, label=false)
scatter(mean(sigma_beta_samples, dims=2)[:, 1])


plt = density(beta_samples' .* sigma_beta_samples', label=false)
ylabel!("Density")
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_posterior_beta.pdf"))

# ------ Mirror Statistic ------

# Monte Carlo loop
mc_samples = 5000
ms_samples = Int(mc_samples / 2)

mean_sigma = MirrorVI.mean(rand(q[prior_position[:sigma_beta]], mc_samples), dims=2)[:, 1]
beta = rand(q[prior_position[:beta]], mc_samples)

ms_coeffs = MirrorStatistic.mirror_statistic(beta[:, 1:ms_samples], beta[:, ms_samples+1:mc_samples])
opt_t = MirrorStatistic.get_t(ms_coeffs; fdr_target=0.1)
inclusion_matrix = ms_coeffs .> opt_t
n_inclusion_per_mc = sum(inclusion_matrix, dims=1)[1, :]
mean_inclusion_per_coef = MirrorVI.mean(inclusion_matrix, dims=2)[:, 1]

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    fdr_target
)

metrics = MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0.,
    selection
)


plt_n = histogram(n_inclusion_per_mc, bins=10, label=false, normalize=true)
xlabel!("# variables included", labelfontsize=15)
vline!([mean(n_inclusion_per_mc)], color = :red, linewidth=5, label="average")
display(plt_n)
savefig(plt_n, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_n_vars_included.pdf"))

plt_probs = scatter(
    findall((1 .- mean_inclusion_per_coef) .> c_opt),
    mean_inclusion_per_coef[findall((1 .- mean_inclusion_per_coef) .> c_opt)],
    label=false, markersize=3,
)
scatter!(
    findall((1 .- mean_inclusion_per_coef) .<= c_opt),
    mean_inclusion_per_coef[findall((1 .- mean_inclusion_per_coef) .<= c_opt)],
    label="Selected", markersize=5,
)
xlabel!("Coefficients", labelfontsize=15)
ylabel!("Inclusion Probability", labelfontsize=15)
vspan!(plt_probs, [p0+1, p], color = :green, alpha = 0.2, labels = "true active coefficients")
display(plt_probs)
savefig(plt_probs, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_mean_selection_matrix.pdf"))

plt = plot(plt_n, plt_probs)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_n_and_probs.pdf"))

scatter(mean_sigma)

scatter(ms_coeffs[:, 1:100], label=false)
hline!([-opt_t, opt_t], linewidth=3)

L = vec(sum(ms_coeffs .< -opt_t, dims=2))
R = vec(sum(ms_coeffs .> opt_t, dims=2))

scatter(R)
scatter!(L)
sum((L ./ R) .< 0.1)

scatter(R)
scatter(R / ms_samples)
hline!([0.1])
selection = R / ms_samples .>= fdr_target

scatter(L)
scatter(L / ms_samples)
hline!([0.1])

ratio = L ./ R
selection = ratio .<= fdr_target
scatter(ratio)
hline!([0.1])

MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0.,
    selection
)


# Loop with relative frequencies
# B-FDR, Monte Carlo approximation
MirrorVI.Random.seed!(134)
mc_samples = 1000

fdp = []
fdp_estimated = []
fp_estimated = []
sum_included = []
matrix_mirror_coeff = zeros(p, mc_samples)
thresholds = []
matrix_included = zeros(p, mc_samples)

for mc = 1:mc_samples
    beta = rand(q[prior_position[:beta]], 2) .* mean_sigma
    mirror_coeff = MirrorStatistic.mirror_statistic(beta[:, 1], beta[:, 2])
    matrix_mirror_coeff[:, mc] = mirror_coeff

    opt_t = MirrorStatistic.get_t(mirror_coeff; fdr_target=fdr_target)
    push!(thresholds, opt_t)
    push!(sum_included, sum(mirror_coeff .> opt_t))
    push!(fp_estimated, sum(mirror_coeff .< -opt_t))

    push!(fdp_estimated, sum(mirror_coeff .< -opt_t) / sum(mirror_coeff .> opt_t))
    matrix_included[:, mc] = mirror_coeff .> opt_t

    push!(fdp,
        MirrorStatistic.false_discovery_rate(
            true_coef=data_dict["beta"] .!= 0.,
            estimated_coef=mirror_coeff .> opt_t
        )
    )
end

mean(fdp)
histogram(fdp)

histogram(sum_included)
mean(sum_included)

# relative inclusion frequency
selection = MirrorStatistic.relative_inclusion_frequency(matrix_included, 0.1, "max")
MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0,
    selection
)


mean_ms = mean(matrix_mirror_coeff, dims=2)[:, 1]
scatter(mean_ms)
opt_t = MirrorStatistic.get_t(mean_ms; fdr_target=fdr_target)
MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0,
    mean_ms .> opt_t
)
sum(mean_ms .> opt_t)

# -------------------------------------------------------
# Knockoffs 
using GLMNet

fdr = []
tpr = []

for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = MirrorVI.generate_linear_model_data(
        n_individuals=n_individuals,
        p=p, p1=p1, p0=p0, corr_factor=corr_factor,
        random_seed=random_seed + simu
    )

    Xk_0 = rand(MirrorVI.MultivariateNormal(data_dict["cov_matrix_0"]), n_individuals)
    Xk_1 = rand(MirrorVI.MultivariateNormal(data_dict["cov_matrix_1"]), n_individuals)
    Xk = transpose(vcat(Xk_0, Xk_1))

    X_aug = hcat(data_dict["X"], Xk)

    glm_k = GLMNet.glmnetcv(X_aug, Float64.(data_dict["y"]))
    coefs = GLMNet.coef(glm_k)

    beta_1 = coefs[1:p]
    beta_2 = coefs[p+1:2*p]

    mirror_coeffs = abs.(beta_1) .- abs.(beta_2)
    
    opt_t = MirrorStatistic.get_t(mirror_coeffs; fdr_target=fdr_target)
    n_selected = sum(mirror_coeffs .> opt_t)

    metrics = MirrorStatistic.wrapper_metrics(
        data_dict["beta"] .!= 0.,
        mirror_coeffs .> opt_t
    )

    push!(fdr, metrics.fdr)
    push!(tpr, metrics.tpr)

end

plt = violin([1], fdr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([1], fdr, label=false, color="blue", fillalpha=0.1, linewidth=2)

violin!([2], tpr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([2], tpr, label=false, color="blue", fillalpha=0.1, linewidth=2)

xticks!([1, 2], ["FDR", "TPR"], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot_Knock.pdf"))


# All plots together
# ---------- FDR ----------
# MC
plt = violin([1], posterior_fdr, color="lightblue", label="BayesMS", alpha=1, linewidth=0)
boxplot!([1], posterior_fdr, label=false, linecolor="blue", color="blue", fillalpha=0., linewidth=2)
# Knockoffs
violin!([2], fdr, color="lightgreen", label="Knockoff", alpha=1, linewidth=0)
boxplot!([2], fdr, label=false, linecolor="green", color="green", fillalpha=0., linewidth=2)
# DS
violin!([3], ds_fdr, color="lightgrey", label="DS", alpha=1, linewidth=0)
boxplot!([3], ds_fdr, label=false, linecolor="grey", color="grey", fillalpha=0., linewidth=2)

# ---------- TPR ----------
# MC
violin!([5], posterior_tpr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([5], posterior_tpr, label=false, linecolor="blue", color="blue", fillalpha=0., linewidth=2)
# Knockoffs
violin!([6], tpr, color="lightgreen", label=false, alpha=1, linewidth=0)
boxplot!([6], tpr, label=false, linecolor="green", color="green", fillalpha=0., linewidth=2)
# DS
violin!([7], ds_tpr, color="lightgrey", label=false, alpha=1, linewidth=0)
boxplot!([7], ds_tpr, label=false, linecolor="grey", color="grey", fillalpha=0., linewidth=2)

xticks!([1, 2, 3, 5, 6, 7], ["", "FDR", "", "", "TPR", ""], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)


savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot_R.pdf"))

# Simulations logistic model
using MirrorVI
using MirrorVI: update_parameters_dict, DistributionsLogPdf, VariationalDistributions, LogExpFunctions, Predictors, mean

using CSV
using DataFrames
using StatsPlots

abs_project_path = normpath(joinpath(@__FILE__, ".."))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))


n_individuals = 500

p = 100
prop_non_zero = 0.1
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5

num_iter = 1000
MC_SAMPLES = 10000
fdr_target = 0.1
n_simulations = 30
random_seed = 125
simulations_models = Dict()
simulations_metrics = Dict()

label_files = "algo_logistic_n$(n_individuals)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"

# Define priors and Variational Distributions
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
    logpdf_prior=x::Real -> DistributionsLogPdf.log_half_cauchy(x, sigma=3.),
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

prior_position = params_dict["tuple_prior_position"]

function model(theta::MirrorVI.ComponentArray; X::AbstractArray)
    beta_reg = theta[:beta] .* theta[:sigma_beta]
    mu = X * beta_reg .+ theta[:beta0]
    return (mu, )
end


for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = MirrorVI.generate_logistic_model_data(;
        n_individuals, class_threshold=0.5,
        p, p1, p0, beta_pool=[-2, -1, 1, 2], obs_noise_sd=0.5, corr_factor=0.5,
        random_seed=random_seed + simu, dtype=Float64
    )
    
    # Training
    z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
    # optimiser = MyOptimisers.DecayedADAGrad()
    optimiser = MirrorVI.Optimisers.RMSProp(0.01)
    
    res = MirrorVI.hybrid_training_loop(
        z=z,
        y=data_dict["y"],
        X=data_dict["X"],
        params_dict=params_dict,
        model=model,
        log_likelihood=DistributionsLogPdf.log_bernoulli_from_logit,
        log_prior=x::AbstractArray -> MirrorVI.compute_logpdf_prior(x; params_dict=params_dict),
        n_iter=num_iter,
        optimiser=optimiser,
        save_all=false,
        use_noisy_grads=false,
        elbo_samples=2
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
    mean_sigma = mean(rand(q[prior_position[:sigma_beta]], mc_samples), dims=2)[:, 1]
    beta = rand(q[prior_position[:beta]], mc_samples)

    ms_coeffs = MirrorStatistic.mirror_statistic(beta[:, 1:ms_samples], beta[:, ms_samples+1:mc_samples])
    opt_t = MirrorStatistic.get_t(ms_coeffs; fdr_target=0.1)
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

mean(posterior_fdr)
mean(mc_fdr)

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


# --------------------------------------------------------------------
# Single Run of Bayesian Model
n = n_individuals * 2
data_dict = MirrorVI.generate_logistic_model_data(;
    n_individuals=n, class_threshold=0.5f0,
    p, p1, p0, beta_pool=Float32.([-2., 2]), obs_noise_sd=0.5, corr_factor=0.5,
    random_seed=124, dtype=Float32
)

n_train = Int(n / 2)
n_test = n - n_train
train_ids = MirrorVI.sample(1:n, n_train, replace=false)
test_ids = setdiff(1:n, train_ids)

X_train = data_dict["X"][train_ids, :]
X_test = data_dict["X"][test_ids, :]
y_train = data_dict["y"][train_ids]
y_test = data_dict["y"][test_ids]


function model(theta::MirrorVI.ComponentArray; X::AbstractArray)
    beta_reg = theta[:beta] .* theta[:sigma_beta]
    mu = X * beta_reg .+ theta[:beta0]
    return (mu, )
end

# Training
z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
optimiser = MyOptimisers.DecayedADAGrad()
optimiser = MirrorVI.Optimisers.RMSProp(0.01)

res = MirrorVI.hybrid_training_loop(
    z=z,
    y=y_train,
    X=X_train,
    params_dict=params_dict,
    model=model,
    log_likelihood=DistributionsLogPdf.log_bernoulli_from_logit,
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
plot(res["loss_dict"]["loss"][500:end])

# Get VI distribution
res["best_iter_dict"]["best_iter"]

beta_samples = rand(q[prior_position[:beta]], MC_SAMPLES)
sigma_samples = rand(q[prior_position[:sigma_beta]], MC_SAMPLES)

density(beta_samples', label=false)
density(beta_samples' .* sigma_samples', label=false)

plt = density(beta_samples' .* sigma_samples', label=false)
ylabel!("Density")
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_posterior_beta.pdf"))


ms_dist = MirrorStatistic.posterior_ms_coefficients(
    q[prior_position[:beta]].dist
)

metrics = MirrorStatistic.optimal_inclusion(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=data_dict["beta"],
    fdr_target=fdr_target
)

# distribution
plt_n = histogram(metrics.n_inclusion_per_mc, bins=10, label=false, normalize=true)
xlabel!("# variables included", labelfontsize=15)
vline!([mean(metrics.n_inclusion_per_mc)], color = :red, linewidth=5, label="average")
display(plt_n)
savefig(plt_n, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_n_vars_included.pdf"))

n_inclusion_per_coef = sum(metrics.inclusion_matrix, dims=2)[:,1]
mean_inclusion_per_coef = mean(metrics.inclusion_matrix, dims=2)[:,1]

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    fdr_target
)

metrics = MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0.,
    selection
)

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


# Monte Carlo loop
mc_samples = 10000
ms_samples = Int(mc_samples / 2)
mean_sigma = mean(rand(q[prior_position[:sigma_beta]], mc_samples), dims=2)[:, 1]
scatter(mean_sigma)
beta = rand(q[prior_position[:beta]], mc_samples) .* mean_sigma
density(beta')

ms_coeffs = MirrorStatistic.mirror_statistic(beta[:, 1:ms_samples], beta[:, ms_samples+1:mc_samples])
opt_t = MirrorStatistic.get_t(ms_coeffs; fdr_target=0.1)
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
sum(selection)

R = vec(sum(ms_coeffs .> opt_t, dims=2))
L = vec(sum(ms_coeffs .< -opt_t, dims=2))
scatter(L ./ R)

selection = R / ms_samples .>= fdr_target
metrics_R = MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0.,
    selection
)
scatter(R / ms_samples)
hline!([0.1])

density(beta[selection, 1:5000]', label=false)
sum(abs.(mean(beta[selection, 1:5000], dims=2)) .> opt_t)


#### GLMNet #####
using GLMNet

y_string = string.(y_train)

glm = GLMNet.glmnetcv(X_train, y_string)
coefs = GLMNet.coef(glm)
sum(coefs .!= 0)
scatter(coefs)


# -------------------------------------------------------
# Knockoffs
using GLMNet
n_simulations = 30

fdr = []
tpr = []

for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = MirrorVI.generate_logistic_model_data(;
        n_individuals, class_threshold=0.5f0,
        p, p1, p0, beta_pool=Float32.([-2., -1, 1, 2]), obs_noise_sd=0.5, corr_factor=0.5,
        random_seed=random_seed + simu, dtype=Float32
    )

    Xk_0 = rand(MirrorVI.MultivariateNormal(data_dict["cov_matrix_0"]), n_individuals)
    Xk_1 = rand(MirrorVI.MultivariateNormal(data_dict["cov_matrix_1"]), n_individuals)
    Xk = transpose(vcat(Xk_0, Xk_1))

    X_aug = hcat(data_dict["X"], Xk)

    glm_k = GLMNet.glmnetcv(X_aug, string.(data_dict["y"]))
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
mean(fdr)

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
plt = violin([1], mc_fdr, color="lightblue", label="BayesMS", alpha=1, linewidth=0)
boxplot!([1], mc_fdr, label=false, linecolor="blue", color="blue", fillalpha=0., linewidth=2)
# Knockoffs
violin!([2], fdr, color="lightgreen", label="Knockoff", alpha=1, linewidth=0)
boxplot!([2], fdr, label=false, linecolor="green", color="green", fillalpha=0., linewidth=2)

# ---------- TPR ----------
# MC
violin!([4], mc_tpr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([4], mc_tpr, label=false, linecolor="blue", color="blue", fillalpha=0., linewidth=2)
# Knockoffs
violin!([5], tpr, color="lightgreen", label=false, alpha=1, linewidth=0)
boxplot!([5], tpr, label=false, linecolor="green", color="green", fillalpha=0., linewidth=2)

xticks!([1, 1.5, 2, 4, 4.5, 5], ["", "FDR", "", "", "TPR", ""], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)


savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot_R.pdf"))



# model-X Knockoff
using Knockoffs

data_dict = MirrorVI.generate_logistic_model_data(;
    n_individuals,
    p1, p0, beta_pool=Float32.([-2., -1, 1, 2]), corr_factor=0.5,
    random_seed=random_seed, dtype=Float64
)

knockoff_filter = Knockoffs.fit_lasso(
    Float64.(data_dict["y"]), data_dict["X"],
    d=Binomial(),
    method=:maxent, m=1, fdrs=[0.1, 0.3]
)
knockoff_filter.selected[1]
knockoff_filter.selected[2]
knockoff_filter.betas

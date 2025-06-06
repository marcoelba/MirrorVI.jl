using MirrorVI
using MirrorVI: update_parameters_dict, DistributionsLogPdf, VariationalDistributions, LogExpFunctions, Predictors, mean

using CSV
using DataFrames
using StatsPlots

abs_project_path = normpath(joinpath(@__FILE__, ".."))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", "results"))


n_individuals = 300

p = 1000
prop_non_zero = 0.05
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5

num_iter = 1000
MC_SAMPLES = 1000
fdr_target = 0.1

function ()
    
end

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
    logpdf_prior=(x::AbstractArray, tau::Real) -> DistributionsLogPdf.log_half_cauchy(x, ones(p)*tau),
    dim_z=p*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=vcat(randn(p)*0.1, randn(p)*0.1),
    dependency=["tau_beta"]
)
update_parameters_dict(
    params_dict;
    name="beta",
    dim_theta=(p, ),
    logpdf_prior=(x::AbstractArray, sigma::AbstractArray) -> DistributionsLogPdf.log_normal(x, sigma=sigma),
    dim_z=p*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
    init_z=vcat(randn(p)*0.01, randn(p)*0.01),
    dependency=["sigma_beta"]
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


random_seed = 1243
data_dict = MirrorVI.generate_linear_model_data(
    n_individuals=n_individuals,
    p=p, p1=p1, p0=p0, corr_factor=corr_factor,
    random_seed=random_seed
)
y = data_dict["y"]
X = data_dict["X"]
y = y .- mean(y)


function linear_model(
    theta::MirrorVI.ComponentArray;
    X::AbstractArray,
    )
    n = size(X, 1)

    mu = theta[:beta0] .+ X * theta[:beta]
    sigma = Float32.(ones(n)) .* theta[:sigma_y]

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
    n_iter=2000,
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

beta_samples = rand(q[prior_position[:beta]], 500)
sigma_beta_samples = rand(q[prior_position[:sigma_beta]], 500)
tau_samples = rand(q[prior_position[:tau_beta]], 500)

density(beta_samples', label=false)
density(beta_samples' .* sigma_beta_samples', label=false)
density(beta_samples' .* mean(sigma_beta_samples, dims=2)[:, 1]', label=false)
density(tau_samples, label=false)
scatter(mean(sigma_beta_samples, dims=2)[:, 1])


plt = density(beta_samples' .* sigma_beta_samples', label=false)
ylabel!("Density")
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_posterior_beta.pdf"))

# ------ Mirror Statistic ------

# Monte Carlo loop
mc_samples = 2000
ms_samples = Int(mc_samples / 2)

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

fdp = []
for mc = 1:Int(mc_samples/2)

    push!(fdp,
        MirrorStatistic.false_discovery_rate(
            true_coef=data_dict["beta"] .!= 0.,
            estimated_coef=ms_coeffs[:, mc] .> opt_t
        )
    )
end
mean(fdp)


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
    beta = rand(q[prior_position[:beta]], 2)
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
srd_ms = MirrorVI.std(matrix_mirror_coeff, dims=2)[:, 1]

scatter(mean_ms)
opt_t = MirrorStatistic.get_t(mean_ms; fdr_target=fdr_target)
MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0,
    mean_ms .> opt_t
)
sum(mean_ms .> opt_t)

# Experiments on Horseshoe vs product prior
using MirrorVI
using MirrorVI: update_parameters_dict, DistributionsLogPdf, VariationalDistributions, LogExpFunctions, Predictors, MirrorStatistic

using DataFrames
using StatsPlots

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", "results"))


n_individuals = 100

p = 100
prop_non_zero = 0.1
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.

num_iter = 1000
MC_SAMPLES = 1000
fdr_target = 0.1
random_seed = 1234

label_files = "priors_comparison_n$(n_individuals)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"

function plot_trace(res; prior::Symbol, plot_var=true)
    range_z = params_dict["ranges_z"][prior_position[prior]]
    half_range = Int(length(range_z) / 2)

    if length(range_z) > 2
        mean_col = cgrad(:reds, half_range, categorical=true)
        sd_col = cgrad(:greys, half_range, categorical=true)

        plt = plot()
        for (ii, jj) in enumerate(range_z[1]:range_z[1]+half_range-1)
            plot!(
                res["loss_dict"]["z_trace"][:, jj],
                label=false,
                color=mean_col[ii]
            )
        end
        if plot_var
            for (ii, jj) in enumerate(range_z[1]+half_range:range_z[end])
                plot!(
                    res["loss_dict"]["z_trace"][:, jj],
                    label=false,
                    color=sd_col[ii]
                )
            end
        end
    else
        plt = plot(
            res["loss_dict"]["z_trace"][:, range_z[1]],
            label=false,
            color="red"
        )
        if plot_var
            plot!(
                res["loss_dict"]["z_trace"][:, range_z[2]],
                label=false,
                color="grey"
            )
        end
    end

    display(plt)
    
end


function model(
    theta;
    X::AbstractArray,
    prod_prior=true
    )
    if prod_prior
        beta_reg = theta[:sigma_beta] .* theta[:beta]
    else
        beta_reg = theta[:beta]
    end

    mu = theta[:beta0] .+ X * beta_reg
    
    return (mu, )
end


function product_prior(;beta_a=1., beta_b=1., tau_prior=1.)

    params_dict = MirrorVI.OrderedDict()

    # beta 0
    update_parameters_dict(
        params_dict;
        name="beta0",
        dim_theta=(1, ),
        logpdf_prior=x::Real -> DistributionsLogPdf.log_normal(x),
        dim_z=2,
        vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=identity),
        init_z=[0., 0.1],
        can_dropout=false,
        noisy_gradient=0
    )

    # beta fixed
    update_parameters_dict(
        params_dict;
        name="tau_beta",
        dim_theta=(p, ),
        logpdf_prior=x::AbstractArray -> DistributionsLogPdf.log_half_cauchy(x, ones(p) .* tau_prior),
        dim_z=p*2,
        vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.log1pexp),
        init_z=randn(p*2),
        can_dropout=false,
        noisy_gradient=1
    )

    update_parameters_dict(
        params_dict;
        name="sigma_beta",
        dim_theta=(p, ),
        logpdf_prior=x::AbstractArray -> DistributionsLogPdf.log_beta(x, beta_a, beta_b),
        dim_z=p*2,
        vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.logistic),
        init_z=vcat(randn(p)*0.1, randn(p)*0.1),
        dependency=[],
        can_dropout=false,
        noisy_gradient=1
    )
    update_parameters_dict(
        params_dict;
        name="beta",
        dim_theta=(p, ),
        logpdf_prior=(x::AbstractArray, sigma::AbstractArray, tau::AbstractArray) -> DistributionsLogPdf.log_normal(x, sigma=sigma .* tau),
        dim_z=p*2,
        vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
        init_z=vcat(randn(p)*0.01, randn(p)*0.01),
        dependency=["sigma_beta", "tau_beta"],
        can_dropout=true,
        noisy_gradient=1
    )

    return params_dict
end


function horseshoe_prior(;tau_prior=1.)

    params_dict = MirrorVI.OrderedDict()

    # beta 0
    update_parameters_dict(
        params_dict;
        name="beta0",
        dim_theta=(1, ),
        logpdf_prior=x::Real -> DistributionsLogPdf.log_normal(x),
        dim_z=2,
        vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=identity),
        init_z=[0., 0.1],
        can_dropout=false
    )

    # beta fixed
    update_parameters_dict(
        params_dict;
        name="tau_beta",
        dim_theta=(1, ),
        logpdf_prior=x::Real -> DistributionsLogPdf.log_half_cauchy(x, sigma=tau_prior),
        dim_z=2,
        vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp),
        init_z=vcat(0.1, 0.1),
        can_dropout=false
    )

    update_parameters_dict(
        params_dict;
        name="sigma_beta",
        dim_theta=(p, ),
        logpdf_prior=(x::AbstractArray, tau::Real) -> DistributionsLogPdf.log_half_cauchy(x, ones(p) .* tau),
        dim_z=p*2,
        vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.log1pexp),
        init_z=vcat(randn(p)*0.1, randn(p)*0.1),
        dependency=["tau_beta"],
        can_dropout=false
    )
    update_parameters_dict(
        params_dict;
        name="beta",
        dim_theta=(p, ),
        logpdf_prior=(x::AbstractArray, sigma::AbstractArray) -> DistributionsLogPdf.log_normal(x, sigma=sigma),
        dim_z=p*2,
        vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
        init_z=vcat(randn(p)*0.01, randn(p)*0.01),
        dependency=["sigma_beta"],
        can_dropout=true
    )

    return params_dict
end


# data generation
data_dict = MirrorVI.generate_logistic_model_data(
    n_individuals=n_individuals,
    p=p, p1=p1, p0=p0, corr_factor=corr_factor,
    random_seed=random_seed
)


params_dict = product_prior(beta_a=1, beta_b=1, tau_prior=1)
prod_prior = false

params_dict = horseshoe_prior(tau_prior=1)
prod_prior = false

prior_position = params_dict["tuple_prior_position"]

z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
optimiser = MirrorVI.MyOptimisers.DecayedADAGrad()

res = MirrorVI.hybrid_training_loop(
    z=z,
    y=data_dict["y"],
    X=data_dict["X"],
    params_dict=params_dict,
    model=(theta; X) -> model(theta; X, prod_prior=prod_prior),
    log_likelihood=DistributionsLogPdf.log_bernoulli_from_logit,
    log_prior=x::AbstractArray -> MirrorVI.compute_logpdf_prior(x; params_dict=params_dict),
    n_iter=3000,
    optimiser=optimiser,
    save_all=true,
    use_noisy_grads=false,
    elbo_samples=3,
    dropout=false
)

plot(res["loss_dict"]["loss"])
plot(res["loss_dict"]["loss"][1500:end])

# plot z trace per category
plot(res["loss_dict"]["z_trace"], label=false)

plot_trace(res, prior=:sigma_beta, plot_var=false)
plot_trace(res, prior=:beta, plot_var=false)
plot_trace(res, prior=:tau_beta, plot_var=false)
plot_trace(res, prior=:sigma_y, plot_var=false)


res["best_iter_dict"]

z = res["best_iter_dict"]["best_z"]

z = res["best_iter_dict"]["final_z"]

##
# z[params_dict["ranges_z"][prior_position[:beta]]] .*= vcat(zeros(p0), ones(p1), ones(p))
##

q = VariationalDistributions.get_variational_dist(
    z,
    params_dict["vi_family_array"],
    params_dict["ranges_z"]
)

beta_samples = rand(q[prior_position[:beta]], MC_SAMPLES)'
sigma_beta_samples = rand(q[prior_position[:sigma_beta]], MC_SAMPLES)'

# beta
plt = density(beta_samples, label=false)

# product prior
plt = density(beta_samples .* sigma_beta_samples, label=false)

ylabel!("Density")
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_posterior_beta.pdf"))

# lambda values
density(sigma_beta_samples, label=false)
scatter(MirrorVI.mean(sigma_beta_samples, dims=1)', label="Average lambda")


# ------ Mirror Statistic ------

# Monte Carlo loop
mc_samples = 2000
ms_samples = Int(mc_samples / 2)

mean_sigma = MirrorVI.mean(rand(q[prior_position[:sigma_beta]], mc_samples), dims=2)[:, 1]
beta = rand(q[prior_position[:beta]], mc_samples) .* mean_sigma

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

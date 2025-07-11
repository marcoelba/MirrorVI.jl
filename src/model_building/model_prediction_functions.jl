# Model prediction functions

module Predictors
using ComponentArrays


function linear_model(
    theta::ComponentArray;
    X::AbstractArray,
    )
    n = size(X, 1)

    mu = theta[:beta0] .+ X * theta[:beta]
    sigma = Float32.(ones(n)) .* theta[:sigma_y]

    return (mu, sigma)
end


function linear_predictor(
    theta::ComponentArray;
    X::AbstractArray,
    link=identity
    )

    mu = theta[:beta0] .+ X * theta[:beta]
    
    return (link.(mu),)
end


function random_intercept_model(
    theta::ComponentArray,
    rep_index::Int64;
    X::AbstractArray
    )
    n = size(X, 1)

    beta_reg = theta[:sigma_beta] .* theta[:beta_fixed]

    mu = theta[:beta0_fixed] .+ theta[:beta0_random] .+ X[:, :, rep_index] * beta_reg
    sigma = ones(eltype(X), n) .* theta[:sigma_y]

    return (mu, sigma)
end


function linear_time_model(
    theta::ComponentArray;
    X::AbstractArray
    )
    n, p = size(X)
    n_time = length(theta[:beta_time])

    # baseline
    mu_inc = [
        theta[:beta_time][tt] .+ X * theta[:beta_fixed][:, tt] for tt = 1:n_time
    ]

    mu = cumsum(reduce(hcat, mu_inc), dims=2)
    sigma = reduce(hcat, [Float32.(ones(n)) .* theta[:sigma_y] for tt = 1:n_time])

    return (mu, sigma)
end


function linear_time_random_intercept_model(
    theta::ComponentArray,
    rep_index::Int64;
    X::AbstractArray
    )
    n_individuals = size(X, 1)

    beta_time = theta[:beta_time]
    # beta_reg = theta[:sigma_beta] .* theta[:beta_fixed]
    beta_reg = theta[:beta_fixed]
    n_time_points = size(beta_time, 1)

    # baseline
    mu_baseline = beta_time[1, rep_index] .+ theta[:beta0_random] .+ X[:, :, rep_index] * beta_reg[:, 1, rep_index]
    mu_inc = [
        Float32.(ones(n_individuals)) .* beta_time[tt, rep_index] .+ X[:, :, rep_index] * beta_reg[:, tt, rep_index] for tt = 2:n_time_points
    ]
    
    mu_matrix = reduce(hcat, [mu_baseline, reduce(hcat, mu_inc)])

    mu = cumsum(mu_matrix, dims=2)

    sigma = theta[:sigma_y] .* Float32.(ones(n_individuals, n_time_points))
    
    return (mu, sigma)
end


end

# training utils
using ProgressMeter
using Zygote
using Optimisers
using ComponentArrays, UnPack


"""
    polynomial_decay(t::Int64; a::Float32=1f0, b::Float32=0.01f0, gamma::Float32=0.75f0)

Compute the polynomial decay value at step `t` using the formula:
a * (b + t)^(-gamma)

This function is commonly used in optimization algorithms (e.g., learning rate scheduling) to decay a value polynomially over time.

    # Arguments
    - `t::Int64`: The current step or time at which to compute the decay.
    - `a::Float32=1f0`: The initial scaling factor. Default is `1.0`.
    - `b::Float32=0.01f0`: A small constant to avoid division by zero. Default is `0.01`.
    - `gamma::Float32=0.75f0`: The decay rate. Controls how quickly the value decays. Default is `0.75`.
    
    # Returns
    - `Float32`: The decayed value at step `t`.
    
    # Examples
    ```julia
    julia> polynomial_decay(10)  # Default parameters
    0.17782794f0
    
    julia> polynomial_decay(100, a=2.0f0, b=0.1f0, gamma=0.5f0)  # Custom parameters
    0.31622776f0
"""
function polynomial_decay(t::Int64; a::Float32=1f0, b::Float32=0.01f0, gamma::Float32=0.75f0)
    a * (b + t)^(-gamma)
end


"""
    cyclical_polynomial_decay(n_iter::Int64, n_cycles::Int64=2)

Generate a cyclical polynomial decay schedule over `n_iter` iterations, divided into `n_cycles` cycles.

This function creates a learning rate schedule where the polynomial decay is applied cyclically.
    Each cycle consists of `steps_per_cycle = n_iter / n_cycles` steps, and the `polynomial_decay` function is applied within each cycle.

# Arguments
- `n_iter::Int64`: The total number of iterations for the schedule.
- `n_cycles::Int64=2`: The number of cycles to divide the iterations into. Default is `2`.

# Returns
- `Vector{Float32}`: A vector of decayed values representing the learning rate schedule.

# Examples
```julia
julia> schedule = cyclical_polynomial_decay(10, 2)  # 10 iterations, 2 cycles
10-element Vector{Float32}:
 0.17782794
 0.125
 0.09765625
 0.080566406
 0.06871948
 0.17782794
 0.125
 0.09765625
 0.080566406
 0.06871948

julia> length(schedule)  # Total number of iterations
10
"""
function cyclical_polynomial_decay(n_iter::Int64, n_cycles::Int64=2)
    steps_per_cycle = Int(n_iter / n_cycles)
    lr_schedule = []
    for cycle = 1:n_cycles
        push!(lr_schedule, polynomial_decay.(range(1, steps_per_cycle))...)
    end
    return lr_schedule
end


"""
    compute_logpdf_prior(theta::ComponentArray; params_dict::OrderedDict)

Compute the sum of the log-pdf of the prior distribution.

# Arguments
- `theta::ComponentArray`: ComponentArray with the components of the parameters sample, one component for each prior.
- `params_dict::OrderedDict`: OrderedDict generated using the MirrorVI.utils functions containing the prior details.

# Returns
- `Real`: The sum of the log-pdf components.
"""
function compute_logpdf_prior(theta::ComponentArray; params_dict::OrderedDict)
    log_prior = 0f0
    priors = params_dict["priors"]

    for (ii, prior) in enumerate(keys(priors))
        deps = priors[prior]["dependency"]
        log_prior += sum(priors[prior]["logpdf_prior"](
            theta[Symbol(prior)],
            [theta[Symbol(dep)] for dep in deps]...
        ))
    end

    return log_prior
end


"""
    get_parameters_axes(params_dict::OrderedDict)

Generate the parameters axes as needed by the library ComponentArrays.
This function processes a dictionary of parameters (`params_dict`) and constructs a prototype array with the same structure as the parameters.
The axes of this prototype array are returned, which can be used to initialize a `ComponentArray` with the correct structure.

# Arguments
- `params_dict::OrderedDict`: OrderedDict generated using the MirrorVI.utils functions containing the prior details.

# Returns
- `Tuple`: The axes of the prototype array, which can be used to initialize a `ComponentArray` with the same structure as the parameters.
"""
function get_parameters_axes(params_dict::OrderedDict)
    
    vector_init = []
    for pp in params_dict["priors"].keys

        if prod(params_dict["priors"][pp]["dim_theta"]) > 1
            param_init = ones(params_dict["priors"][pp]["dim_theta"])
        else
            param_init = ones(params_dict["priors"][pp]["dim_theta"])[1]
        end
        push!(vector_init, Symbol(pp) => param_init)
    end

    proto_array = ComponentArray(; vector_init...)
    theta_axes = getaxes(proto_array)

    return theta_axes
end


function rand_z_dropout(params_dict; dtype=Float64)
    z = []

    for prior in keys(params_dict["priors"])
        pp = params_dict["priors"][prior]

        if pp["can_dropout"]
            # for the mean
            append!(z, rand([0., 1.], Int(pp["dim_z"]/2)))
            # for the var
            append!(z, ones(Int(pp["dim_z"]/2)))
        else
            append!(z, ones(pp["dim_z"]))
        end
    end

    return dtype.(z)
end


"""
    elbo(
        z::AbstractArray;
        y::AbstractArray,
        X::AbstractArray,
        ranges_z::AbstractArray,
        vi_family_array::AbstractArray,
        random_weights::AbstractArray,
        model,
        theta_axes::Tuple,
        log_likelihood,
        log_prior=zero,
        n_samples::Int64=1,
        n_repeated_measures::Int64=1
    )

Compute the Evidence Lower Bound (ELBO) for a variational inference problem.

The ELBO is a key quantity in variational inference, used to approximate the posterior distribution of model parameters.
This function computes the ELBO by:
1. Sampling from the variational distribution.
2. Evaluating the log-likelihood and log-prior of the model.
    Optional: Takes into account repeated measurements, if it is required
3. Adding the entropy of the variational distribution.


# Arguments
- `z::AbstractArray`: The variational parameters used to define the variational distribution.
- `y::AbstractArray`: The observed data (target values).
- `X::AbstractArray`: The input data (features).
- `ranges_z::AbstractArray`: Specifies how `z` is divided among the parameters of the variational distribution.
- `vi_family_array::AbstractArray`: An array of functions defining the variational family for each parameter.
- `random_weights::AbstractArray`: Boolean array of the same dimension as theta, stating whether each parameter is random or not.
- `model`: A function representing the model, which takes parameters and input data `X` (keyword argument) and returns atuple with the predictions.
- `theta_axes::ComponentArrays.Axes`: The axes for constructing a `ComponentArray` from the sampled parameters.
- `log_likelihood`: A function that computes the log-likelihood of the observed data given the model predictions.
- `log_prior=zero`: A function that computes the log-prior of the parameters. Default is `zero` (no prior).
- `n_samples::Int64=1`: The number of Monte Carlo samples to use for approximating the ELBO. Default is `1`.
- `n_repeated_measures::Int64=1`: The number of repeated measurements (e.g., for longitudinal data). Default is `1`.

# Returns
- `Float64`: The negative ELBO value (to be minimized).
"""
function elbo(
    z::AbstractArray;
    y::AbstractArray,
    X::AbstractArray,
    ranges_z::AbstractArray,
    vi_family_array::AbstractArray,
    random_weights::AbstractArray,
    model,
    theta_axes::Tuple,
    log_likelihood,
    log_prior=zero,
    n_samples::Int64=1,
    n_repeated_measures::Int64=1
    )

    # get a specific distribution using the weights z, from the variational family
    q_dist_array = VariationalDistributions.get_variational_dist(z, vi_family_array, ranges_z)
    ndims_y = ndims(y)

    # evaluate the log-joint
    res = zero(eltype(z))

    for mc = 1:n_samples
        
        # random sample from VI distribution
        theta, abs_jacobian = VariationalDistributions.rand_with_logjacobian(q_dist_array, random_weights=random_weights)
        theta_components = ComponentArray(theta, theta_axes)

        # evaluate the log-joint
        loglik = 0.
        if n_repeated_measures == 1
            pred = model(theta_components; X=X)
            loglik += sum(log_likelihood(y, pred...))
        else
            for measurement = 1:n_repeated_measures
                pred = model(theta_components, measurement; X=X)
                loglik += sum(log_likelihood(selectdim(y, ndims_y, measurement), pred...))
            end
        end
        logprior = sum(log_prior(theta_components))

        res += (loglik + logprior + sum(abs_jacobian)) / n_samples
    end

    # add entropy
    for d in q_dist_array
        res += VariationalDistributions.entropy(d)
    end

    return -res
end


"""
    hybrid_training_loop(;
        z::AbstractArray,
        y::AbstractArray,
        X::AbstractArray,
        params_dict::OrderedDict,
        model,
        log_likelihood,
        log_prior=zero,
        n_iter::Int64,
        optimiser::Optimisers.AbstractRule,
        save_all::Bool=false,
        use_noisy_grads::Bool=false,
        elbo_samples::Int64=1,
        lr_schedule=nothing,
        n_repeated_measures::Int64=1,
        dropout::Bool=false,
        start_dropout_iter::Int=0
    )

Run a training loop for variational inference, combining gradient-based optimization with optional noise injection and dropout (experimental).

This function performs variational inference by minimizing the Evidence Lower Bound (ELBO) using gradient-based optimization. It supports:
- Noisy gradients for exploration.
- Dropout for regularization.
- Cyclical learning rate schedules.
- Saving intermediate results for analysis.

# Arguments
- `z::AbstractArray`: Initial variational parameters.
- `y::AbstractArray`: Observed data (target values).
- `X::AbstractArray`: Input data (features).
- `params_dict::OrderedDict`: Ordered Dictionary defined through MirrorVI.utils functions, containing configuration for the variational family, parameter ranges, and other settings. Must include:
  - `"vi_family_array"`: Array of functions defining the variational family for each parameter.
  - `"ranges_z"`: Specifies how `z` is divided among the parameters of the variational distribution.
  - `"random_weights"`: Weights used for sampling from the variational distribution.
  - `"noisy_gradients"`: Standard deviation of noise added to gradients (if `use_noisy_grads=true`).
- `model`: A function representing the model, which takes parameters and input data `X` and returns predictions.
- `log_likelihood`: A function that computes the log-likelihood of the observed data given the model predictions.
- `log_prior=zero`: A function that computes the log-prior of the parameters. Default is `zero` (no prior).
- `n_iter::Int64`: Number of training iterations.
- `optimiser::Optimisers.AbstractRule`: Optimiser to use for updating `z` (e.g., `DecayedADAGrad`).
- `save_all::Bool=false`: If `true`, saves the trace of `z` across all iterations. Default is `false`.
- `use_noisy_grads::Bool=false`: If `true`, adds noise to the gradients during training. Default is `false`.
- `elbo_samples::Int64=1`: Number of Monte Carlo samples to use for approximating the ELBO. Default is `1`.
- `lr_schedule=nothing`: Learning rate schedule (e.g., from `cyclical_polynomial_decay`). Default is `nothing`.
- `n_repeated_measures::Int64=1`: Number of repeated measurements (e.g., for time-series data). Default is `1`.
- `dropout::Bool=false`: If `true`, applies dropout to `z` during training. Default is `false`.
- `start_dropout_iter::Int=0`: Iteration at which to start applying dropout. Default is `0`.

# Returns
- `Dict`: A dictionary containing:
  - `"loss_dict"`: A dictionary with keys:
    - `"loss"`: Array of ELBO values across iterations.
    - `"z_trace"`: Trace of `z` across iterations (if `save_all=true`).
  - `"best_iter_dict"`: A dictionary with keys:
    - `"best_loss"`: Best ELBO value achieved.
    - `"best_z"`: Variational parameters corresponding to the best ELBO.
    - `"final_z"`: Final variational parameters after training.
    - `"best_iter"`: Iteration at which the best ELBO was achieved.

"""
function hybrid_training_loop(;
    z::AbstractArray,
    y::AbstractArray,
    X::AbstractArray,
    params_dict::OrderedDict,
    model,
    log_likelihood,
    log_prior=zero,
    n_iter::Int64,
    optimiser::Optimisers.AbstractRule,
    save_all::Bool=false,
    use_noisy_grads::Bool=false,
    elbo_samples::Int64=1,
    lr_schedule=nothing,
    n_repeated_measures::Int64=1,
    dropout::Bool=false,
    start_dropout_iter::Int=0
    )

    vi_family_array = params_dict["vi_family_array"]
    ranges_z = params_dict["ranges_z"]
    random_weights = params_dict["random_weights"]
    noisy_gradients = params_dict["noisy_gradients"]
    theta_axes = get_parameters_axes(params_dict)

    # store best setting
    best_z = copy(z)
    best_loss = eltype(z)(Inf)
    best_iter = 1

    # lr_schedule = cyclical_polynomial_decay(n_iter, n_cycles)
    state = Optimisers.setup(optimiser, z)

    prog = ProgressMeter.Progress(n_iter, 1)
    iter = 1

    # store results
    if save_all
        z_trace = zeros(eltype(z), n_iter, length(z))
    else
        z_trace = nothing
    end
    loss = []
    
    while iter ≤ n_iter
        
        train_loss, grads = Zygote.withgradient(z) do zp
            elbo(zp;
                y=y,
                X=X,
                ranges_z=ranges_z,
                vi_family_array=vi_family_array,
                random_weights=random_weights,
                model=model,
                theta_axes=theta_axes,
                log_likelihood=log_likelihood,
                log_prior=log_prior,
                n_samples=elbo_samples,
                n_repeated_measures=n_repeated_measures
            )
        end
    
        # z update
        Optimisers.update!(state, z, grads[1])
        
        # Using noisy gradients?
        if use_noisy_grads
            rand_z = randn(eltype(z), length(z)) .* noisy_gradients .* lr_schedule[iter]
            z .+= rand_z
        end

        if dropout && iter > start_dropout_iter
            rand_drop = rand_z_dropout(params_dict; dtype=eltype(z))
            z .*= rand_drop
        end

        # store
        push!(loss, train_loss)
        if save_all
            z_trace[iter, :] = z
        end
        
        # elbo check
        if train_loss < best_loss
            best_loss = copy(train_loss)
            best_z = copy(z)
            best_iter = copy(iter)
        end

        iter += 1
        ProgressMeter.next!(prog)
    end

    loss_dict = Dict("loss" => loss, "z_trace" => z_trace)

    best_iter_dict = Dict(
        "best_loss" => best_loss,
        "best_z" => best_z,
        "final_z" => z,
        "best_iter" => best_iter
    )

    return Dict("loss_dict" => loss_dict, "best_iter_dict" => best_iter_dict)
end

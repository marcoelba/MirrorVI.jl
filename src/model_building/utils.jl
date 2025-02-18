# Model definition

using OrderedCollections
using StatsFuns
using ComponentArrays, UnPack


function update_parameters_dict(
    params_dict::OrderedDict;
    name::String,
    dim_theta::Tuple,
    logpdf_prior,
    dim_z::Int64,
    vi_family,
    init_z=randn(dim_z),
    dependency=[],
    random_variable::Bool=true,
    noisy_gradient::Int64=0,
    can_dropout::Bool=false
    )

    if !("priors" in keys(params_dict))
        # create sub-dictionary if first call
        params_dict["priors"] = OrderedDict()  
    end

    parameter_already_included = false
    if name in keys(params_dict["priors"])
        @warn "Prior <$(name)> overwritten"
        parameter_already_included = true
    end

    # If first call
    if !("tot_params" in keys(params_dict))
        params_dict["tot_params"] = 0
        params_dict["ranges_theta"] = []

        params_dict["tot_vi_weights"] = 0
        params_dict["ranges_z"] = []
        params_dict["vi_family_array"] = []
        params_dict["reshape_array"] = []

        params_dict["keys_prior_position"] = OrderedDict()
        params_dict["random_weights"] = []
        params_dict["noisy_gradients"] = []
    end

    if !(parameter_already_included)
        
        # first time call #

        # theta
        range_theta = (params_dict["tot_params"] + 1):(params_dict["tot_params"] + prod(dim_theta))
        params_dict["tot_params"] = params_dict["tot_params"] + prod(dim_theta)
        # range VI weights (z)
        range_z = (params_dict["tot_vi_weights"] + 1):(params_dict["tot_vi_weights"] + dim_z)
        params_dict["tot_vi_weights"] = params_dict["tot_vi_weights"] + dim_z

    else
        range_theta = params_dict["priors"][name]["range_theta"]
        params_dict["tot_params"] = params_dict["tot_params"]
        # VI weights
        range_z = params_dict["priors"][name]["range_z"]
        params_dict["tot_vi_weights"] = params_dict["tot_vi_weights"]

    end

    new_prior = OrderedDict(
        "dim_theta" => dim_theta,
        "range_theta" => range_theta,
        "logpdf_prior" => logpdf_prior,
        "dim_z" => dim_z,
        "range_z" => range_z,
        "vi_family" => vi_family,
        "init_z" => init_z,
        "dependency" => dependency,
        "random_variable" => random_variable,
        "can_dropout" => can_dropout
    )

    params_dict["priors"][name] = new_prior

    # Create a tuple for the ranges and the transformations
    if !(parameter_already_included)
        push!(params_dict["ranges_theta"], params_dict["priors"][name]["range_theta"])
        push!(params_dict["ranges_z"], params_dict["priors"][name]["range_z"])
        push!(params_dict["vi_family_array"], params_dict["priors"][name]["vi_family"])
        push!(params_dict["random_weights"], params_dict["priors"][name]["random_variable"])
        append!(params_dict["noisy_gradients"], ones(dim_z) .* noisy_gradient)
        push!(params_dict["reshape_array"], dim_theta)

        params_dict["keys_prior_position"][Symbol(name)] = length(params_dict["vi_family_array"])
        params_dict["tuple_prior_position"] = (; params_dict["keys_prior_position"]...)
    end
    
    return params_dict
end

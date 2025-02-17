# Variational Distributions

module VariationalDistributions
using DistributionsAD
using Bijectors
using StatsFuns
using LogExpFunctions

using MirrorVI: ConstantDistribution, MultivariateConstantDistribution


function get_variational_dist(z::AbstractArray, vi_family_array, ranges_z)
    
    q_vi = [vi_family_array[cc](z[ranges_z[cc]]) for cc in eachindex(vi_family_array)]

    return q_vi
end


function get_init_z(params_dict; dtype=Float64)
    z = []
    for prior in keys(params_dict["priors"])
        append!(z, params_dict["priors"][prior]["init_z"])
    end

    return dtype.(z)
end


function rand_array(q_dist_array::AbstractArray; from_base_dist::Bool=false, reduce_to_vec::Bool=false)
    if from_base_dist
        v_sample = [rand(dist.dist) for dist in q_dist_array]
    else
        v_sample = [rand(dist) for dist in q_dist_array]
    end

    if reduce_to_vec
        v_sample = reduce(vcat, v_sample)
    end

    return v_sample
end


function rand_with_logjacobian(q_dist_array::AbstractArray; random_weights::AbstractArray)
    x = rand_array(q_dist_array, from_base_dist=true, reduce_to_vec=false)
    x_t = [q_dist_array[ii].transform(x[ii]) for ii in eachindex(x)]

    abs_jacobian = zero(eltype(x[1]))
    for ii in eachindex(x)
        abs_jacobian += random_weights[ii] * Bijectors.jacobian(q_dist_array[ii].transform, x[ii], x_t[ii])
    end

    return reduce(vcat, x_t), abs_jacobian
end


# Entropy
function entropy(d::Bijectors.TransformedDistribution)
    DistributionsAD.entropy(d.dist)
end


function meanfield(z::AbstractArray{Float32}; tot_params::Int64)
    DistributionsAD.MultivariateNormal(
        z[1:tot_params],
        LogExpFunctions.softplus.(z[(tot_params + 1):(tot_params * 2)])
    )
end


function vi_mv_normal(z::AbstractArray; bij=identity)
    zdim = length(z)
    pdim = Int(zdim / 2)

    Bijectors.transformed(
        DistributionsAD.MvNormal(
            z[1:pdim],
            LogExpFunctions.log1pexp.(z[pdim+1:zdim])
        ),
        bij
    )
end

function vi_mv_normal_known_var(z::AbstractArray; bij=identity)
    zdim = length(z)

    Bijectors.transformed(
        DistributionsAD.MvNormal(
            z,
            ones(eltype(z), zdim)
        ),
        bij
    )
end


function vi_constant_mv_normal(z::AbstractArray; bij=identity)
    Bijectors.transformed(
        MultivariateConstantDistribution(
            z
        ),
        bij
    )
end


function vi_normal(z::AbstractArray; bij=identity)
    Bijectors.transformed(
        DistributionsAD.Normal(
            z[1],
            LogExpFunctions.log1pexp.(z[2])
        ),
        bij
    )
end

function vi_constant_normal(z::AbstractArray; bij=identity)
    Bijectors.transformed(
        ConstantDistribution(z[1]),
        bij
    )
end


end

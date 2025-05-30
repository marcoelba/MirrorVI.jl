# Mirror Statistic

module MirrorStatistic

using Distributions
using OrderedCollections
using LinearAlgebra


function mean_folded_normal(mu, sigma)
    sigma * sqrt(2/pi) * exp(-0.5 *(mu/sigma)^2) + mu * (1 - 2*cdf(Normal(), -(mu/sigma)))
end

function var_folded_normal(mu, sigma)
    mu^2 + sigma^2 - mean_folded_normal(mu, sigma)^2
end


function posterior_ms_coefficients(vi_posterior::Distributions.DiagNormal)

    # get mean and std of the multivariate normal on the regression coefficients
    mu = vi_posterior.μ
    sigma = sqrt.(diag(vi_posterior.Σ))
    
    # MS Distribution
    mean_vec = MirrorStatistic.mean_folded_normal.(mu, sigma) .- 
        MirrorStatistic.mean_folded_normal.(0., sigma)

    var_vec = MirrorStatistic.var_folded_normal.(mu, sigma) .+ 
        MirrorStatistic.var_folded_normal.(0., sigma)

    ms_dist_vec = [
        Distributions.Normal(mean_ms, sqrt(var_ms)) for (mean_ms, var_ms) in zip(mean_vec, var_vec)
    ]

    return ms_dist_vec
end


function get_t(mirror_coeffs; fdr_target)
    
    optimal_t = 0
    t = 0
    
    for t in range(0., maximum(mirror_coeffs), length=2000)
        n_left_tail = sum(mirror_coeffs .< -t)
        n_right_tail = sum(mirror_coeffs .> t)
        n_right_tail = ifelse(n_right_tail .> 0, n_right_tail, 1)
    
        fdp = n_left_tail / n_right_tail
    
        if fdp <= fdr_target
            optimal_t = t
            break
        end
    end
    return optimal_t
end


function mirror_statistic(theta_1, theta_2)
    abs.(theta_1 .+ theta_2) .- abs.(theta_1 .- theta_2)
end


function optimal_inclusion(;ms_dist_vec, mc_samples::Int64, beta_true, fdr_target::Real=0.1)

    p = length(beta_true)
    mirror_coefficients = reduce(hcat, rand.(ms_dist_vec, mc_samples))'
    opt_t = get_t(mirror_coefficients; fdr_target=fdr_target)

    inclusion_matrix = mirror_coefficients .> opt_t
    n_inclusion_per_mc = sum(inclusion_matrix, dims=1)[1,:]
    sort_indices = sortperm(sum(mirror_coefficients .> opt_t, dims=2)[:,1], rev=true)

    fdr_range = []
    tpr_range = []
    for n in range(minimum(n_inclusion_per_mc), maximum(n_inclusion_per_mc))
        selection = zeros(p)
        selection[sort_indices[1:n]] .= 1
        metrics = wrapper_metrics(
            beta_true .!= 0.,
            selection .> 0
        )
        push!(fdr_range, metrics.fdr)
        push!(tpr_range, metrics.tpr)

    end

    selection = zeros(p)
    selection[sort_indices[1:Int(round(mean(n_inclusion_per_mc)))]] .= 1
    metrics_mean = wrapper_metrics(
        beta_true .!= 0.,
        selection .> 0
    )
    selection = zeros(p)
    selection[sort_indices[1:Int(round(median(n_inclusion_per_mc)))]] .= 1
    metrics_median = wrapper_metrics(
        beta_true .!= 0.,
        selection .> 0
    )

    # get FDR distribution over all MC samples
    fdr_distribution = zeros(mc_samples)
    tpr_distribution = zeros(mc_samples)

    for nn = 1:mc_samples
        metrics = wrapper_metrics(
            beta_true .!= 0.,
            inclusion_matrix[:, nn]
        )
        fdr_distribution[nn] = metrics.fdr
        tpr_distribution[nn] = metrics.tpr
    end
    
    return (
        fdr_range=fdr_range,
        tpr_range=tpr_range,
        fdr_distribution=fdr_distribution,
        tpr_distribution=tpr_distribution,
        metrics_mean=metrics_mean,
        metrics_median=metrics_median,
        inclusion_matrix=inclusion_matrix,
        n_inclusion_per_mc=n_inclusion_per_mc,
        opt_t=opt_t
    )
end


# Newton method to control the FDR from posterior distribution
function posterior_fdr_threshold(inclusion_probs, fdr_target=0.1)
    fp_prob = 1. .- inclusion_probs
    c_opt = 0.

    for c in sort(fp_prob, rev=true)
        lower_than_c = fp_prob .<= c
        if (sum(fp_prob[lower_than_c]) / sum(lower_than_c)) < fdr_target
            c_opt = c
            break
        end
    end
    return (c_opt=c_opt, selection=fp_prob .<= c_opt)
end


function posterior_ms_inclusion(;ms_dist_vec, mc_samples::Int64, beta_true, fdr_target::Real=0.1)

    mirror_coefficients = reduce(hcat, rand.(ms_dist_vec, mc_samples))'
    opt_t = get_t(mirror_coefficients; fdr_target=fdr_target)
    inclusion_matrix = mirror_coefficients .> opt_t
    inclusion_probs = mean(inclusion_matrix, dims=2)[:, 1]

    c_opt, selection = posterior_fdr_threshold(inclusion_probs, fdr_target)

    metrics = wrapper_metrics(
        beta_true .!= 0.,
        selection
    )

    return metrics

end


# classification metrics

function false_discovery_rate(;
    true_coef::AbstractArray,
    estimated_coef::AbstractArray
    )

    sum_coef = true_coef + estimated_coef
    TP = sum(sum_coef .== 2.)
    FP = sum((sum_coef .== 1.) .& (estimated_coef .== 1.))

    tot_predicted_positive = TP + FP

    if tot_predicted_positive > 0
        FDR = FP / tot_predicted_positive
    else
        FDR = 0.
        # println("Warning: 0 Positive predictions")
    end

    return FDR
end


function true_positive_rate(;
    true_coef::Union{Array{Real}, BitVector},
    estimated_coef::Union{Array{Real}, BitVector}
    )

    sum_coef = true_coef + estimated_coef
    TP = sum(sum_coef .== 2.)
    FN = sum((sum_coef .== 1.) .& (true_coef .== 1))

    TPR = TP / (TP + FN)

    return TPR
end


function wrapper_metrics(true_coef, pred_coef)
    # FDR
    fdr = false_discovery_rate(
        true_coef=true_coef,
        estimated_coef=pred_coef
    )

    # TPR
    tpr = true_positive_rate(
        true_coef=true_coef,
        estimated_coef=pred_coef
    )

    return (fdr=fdr, tpr=tpr)

end

"""
    bh_correction(;p_values::Vector{Float64}, fdr_level::Float64)
    
    False Discovery Rate adjusted p-values using Benjamini-Hochberg procedure
"""
function bh_correction(;p_values::AbstractArray, fdr_level::Real)
    if fdr_level > 1.
        throw(error("fdr_level MUST be <= 1"))
    end

    # store the ranks in the original order
    pvalues_ranks_original = hcat(p_values, invperm(sortperm(p_values)))

    n_tests = length(p_values)
    sorted_p_values = hcat(sort(p_values), sortperm(sort(p_values)))
    adjusted_pvalues = (sorted_p_values[:, 2] ./ n_tests) .* fdr_level
    sorted_p_values = hcat(sorted_p_values, adjusted_pvalues)

    # find the largest p-value that is lower than its corresponding BH critical value
    ranks_significant_pvalues = sorted_p_values[sorted_p_values[:, 1] .< sorted_p_values[:, 3], 2]
    pvalues_ranks_original = hcat(pvalues_ranks_original, zeros(n_tests))
    rank = 1
    for rank in range(1, n_tests)
        pvalues_ranks_original[rank, 3] = ifelse(pvalues_ranks_original[rank, 2] in ranks_significant_pvalues, 1, 0)
    end

    return pvalues_ranks_original
end

end
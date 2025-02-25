using Test
using Distributions
using LinearAlgebra
using OrderedCollections

using MirrorVI: MirrorStatistic


@testset "MirrorStatistic Module Tests" begin

    @testset "mean_folded_normal" begin
        # Test cases for mean_folded_normal
        @test isapprox(MirrorStatistic.mean_folded_normal(0.0, 1.0), sqrt(2/π), atol=1e-6)
        @test isapprox(MirrorStatistic.mean_folded_normal(1.0, 1.0), 1.0 + sqrt(2/π) * exp(-0.5) - 2 * cdf(Normal(), -1.0), atol=1e-6)
    end

    @testset "var_folded_normal" begin
        # Test cases for var_folded_normal
        @test isapprox(MirrorStatistic.var_folded_normal(0.0, 1.0), 1.0 - 2/π, atol=1e-6)
        @test isapprox(MirrorStatistic.var_folded_normal(1.0, 1.0), 1.0 + 1.0 - MirrorStatistic.mean_folded_normal(1.0, 1.0)^2, atol=1e-6)
    end

    @testset "posterior_ms_coefficients" begin
        # Test cases for posterior_ms_coefficients
        mu = [1.0, -1.0]
        sigma = [1.0, 1.0]
        diag_normal = MvNormal(mu, Diagonal(sigma.^2))
        ms_dist_vec = MirrorStatistic.posterior_ms_coefficients(diag_normal)
        
        @test length(ms_dist_vec) == 2
        @test ms_dist_vec[1] isa Normal
        @test ms_dist_vec[2] isa Normal
    end

    @testset "get_t" begin
        # Test cases for get_t
        mirror_coeffs = [-1.0, 0.5, 1.5, 2.0]
        fdr_target = 0.1
        t = MirrorStatistic.get_t(mirror_coeffs, fdr_target=fdr_target)
        
        @test t >= 0.0
        @test t <= maximum(mirror_coeffs)
    end

    @testset "mirror_statistic" begin
        # Test cases for mirror_statistic
        theta_1 = [1.0, -1.0, 0.0]
        theta_2 = [1.0, 1.0, 0.0]
        ms = MirrorStatistic.mirror_statistic(theta_1, theta_2)
        
        @test ms == [2.0, -2.0, 0.0]
    end

    @testset "optimal_inclusion" begin
        # Test cases for optimal_inclusion
        ms_dist_vec = [Normal(1.0, 1.0), Normal(-1.0, 1.0)]
        mc_samples = 1000
        beta_true = [1., 0.]
        fdr_target = 0.1
        
        result = MirrorStatistic.optimal_inclusion(
            ms_dist_vec=ms_dist_vec,
            mc_samples=mc_samples,
            beta_true=beta_true,
            fdr_target=fdr_target
        )
        
        @test result.fdr_range isa Vector
        @test result.tpr_range isa Vector
        @test result.fdr_distribution isa Vector
        @test result.tpr_distribution isa Vector
        @test result.metrics_mean isa NamedTuple
        @test result.metrics_median isa NamedTuple
        @test result.inclusion_matrix isa BitMatrix
        @test result.n_inclusion_per_mc isa Vector
        @test result.opt_t isa Real
    end

    @testset "posterior_fdr_threshold" begin
        # Test cases for posterior_fdr_threshold
        inclusion_probs = [0.1, 0.5, 0.9]
        fdr_target = 0.1
        
        c_opt, selection = MirrorStatistic.posterior_fdr_threshold(inclusion_probs, fdr_target)
        
        @test c_opt isa Real
        @test selection isa BitArray
    end

    @testset "posterior_ms_inclusion" begin
        # Test cases for posterior_ms_inclusion
        ms_dist_vec = [Normal(1.0, 1.0), Normal(-1.0, 1.0)]
        mc_samples = 1000
        beta_true = [1., 0.]
        fdr_target = 0.1
        
        metrics = MirrorStatistic.posterior_ms_inclusion(ms_dist_vec=ms_dist_vec, mc_samples=mc_samples, beta_true=beta_true, fdr_target=fdr_target)
        
        @test metrics isa NamedTuple
        @test metrics.fdr isa Real
        @test metrics.tpr isa Real
    end

    @testset "false_discovery_rate" begin
        # Test cases for false_discovery_rate
        true_coef = [1, 0, 1, 0] .> 0
        estimated_coef = [1, 1, 0, 0] .> 0
        
        fdr = MirrorStatistic.false_discovery_rate(true_coef=true_coef, estimated_coef=estimated_coef)
        
        @test fdr isa Real
        @test fdr == 0.5
    end

    @testset "true_positive_rate" begin
        # Test cases for true_positive_rate
        true_coef = [1, 0, 1, 0] .> 0
        estimated_coef = [1, 1, 0, 0] .> 0
        
        tpr = MirrorStatistic.true_positive_rate(true_coef=true_coef, estimated_coef=estimated_coef)
        
        @test tpr isa Real
        @test tpr == 0.5
    end

    @testset "wrapper_metrics" begin
        # Test cases for wrapper_metrics
        true_coef = [1, 0, 1, 0] .> 0
        estimated_coef = [1, 1, 0, 0] .> 0
        
        metrics = MirrorStatistic.wrapper_metrics(true_coef, estimated_coef)
        
        @test metrics isa NamedTuple
        @test metrics.fdr isa Real
        @test metrics.tpr isa Real
        @test metrics.fdr == 0.5
        @test metrics.tpr == 0.5
    end

end

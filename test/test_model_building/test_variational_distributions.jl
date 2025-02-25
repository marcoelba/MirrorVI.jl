using Test
using DistributionsAD

using MirrorVI: ConstantDistribution, MultivariateConstantDistribution, LogExpFunctions, Bijectors, VariationalDistributions


@testset "VariationalDistributions Module Tests" begin

    # Test get_variational_dist
    @testset "get_variational_dist" begin
        z = [1.0, 2.0, 3.0, 4.0]
        vi_family_array = [VariationalDistributions.vi_normal, VariationalDistributions.vi_mv_normal]
        ranges_z = [1:2, 3:4]
        q_vi = VariationalDistributions.get_variational_dist(z, vi_family_array, ranges_z)
        @test length(q_vi) == 2
        @test q_vi[1] isa Bijectors.TransformedDistribution
        @test q_vi[2] isa Bijectors.TransformedDistribution
    end

    # Test get_init_z
    @testset "get_init_z" begin
        params_dict = Dict(
            "priors" => Dict(
                "prior1" => Dict("init_z" => [1.0, 2.0]),
                "prior2" => Dict("init_z" => [3.0, 4.0])
            )
        )
        z = VariationalDistributions.get_init_z(params_dict)
        @test z == [1.0, 2.0, 3.0, 4.0]
        @test eltype(z) == Float64
    end

    # Test rand_array
    @testset "rand_array" begin
        q_dist_array = [
            Bijectors.transformed(DistributionsAD.Normal(0.0, 1.0)),
            Bijectors.transformed(DistributionsAD.Normal(1.0, 2.0))
        ]
        v_sample = VariationalDistributions.rand_array(q_dist_array)
        @test length(v_sample) == 2
        @test v_sample[1] isa Float64
        @test v_sample[2] isa Float64

        v_sample_reduced = VariationalDistributions.rand_array(q_dist_array, reduce_to_vec=true)
        @test v_sample_reduced isa Vector{Float64}
        @test length(v_sample_reduced) == 2
    end

    # Test rand_with_logjacobian
    @testset "rand_with_logjacobian" begin
        q_dist_array = [
            Bijectors.transformed(DistributionsAD.Normal(0.0, 1.0)),
            Bijectors.transformed(DistributionsAD.Normal(1.0, 2.0))
        ]
        random_weights = [1.0, 1.0]
        x_t, abs_jacobian = VariationalDistributions.rand_with_logjacobian(q_dist_array, random_weights=random_weights)
        @test x_t isa Vector{Float64}
        @test abs_jacobian isa Float64
    end

    # Test entropy for TransformedDistribution
    @testset "entropy for TransformedDistribution" begin
        dist = DistributionsAD.Normal(0.0, 1.0)
        transformed_dist = Bijectors.transformed(dist)
        @test VariationalDistributions.entropy(transformed_dist) == Bijectors.entropy(dist)
    end

    # Test meanfield
    @testset "meanfield" begin
        z = Float32[1.0, 2.0, 3.0, 4.0]
        tot_params = 2
        q = VariationalDistributions.meanfield(z, tot_params=tot_params)
        @test q isa DistributionsAD.MultivariateNormal
        @test length(q.μ) == tot_params
        @test length(q.Σ.diag) == tot_params
    end

    # Test vi_mv_normal
    @testset "vi_mv_normal" begin
        z = [1.0, 2.0, 3.0, 4.0]
        q = VariationalDistributions.vi_mv_normal(z)
        @test q isa Bijectors.TransformedDistribution
        @test length(q.dist.μ) == 2
        @test length(q.dist.Σ.diag) == 2
    end

    # Test vi_mv_normal_known_var
    @testset "vi_mv_normal_known_var" begin
        z = [1.0, 2.0, 3.0, 4.0]
        q = VariationalDistributions.vi_mv_normal_known_var(z)
        @test q isa Bijectors.TransformedDistribution
        @test length(q.dist.μ) == 4
        @test all(q.dist.Σ.diag .== 1.0)
    end

    # Test vi_constant_mv_normal
    @testset "vi_constant_mv_normal" begin
        z = [1.0, 2.0, 3.0, 4.0]
        
        q = VariationalDistributions.vi_constant_mv_normal(z)
        @test q isa Bijectors.TransformedDistribution
        @test q.dist isa MultivariateConstantDistribution
        @test q.dist.m == z
    end

    # Test vi_normal
    @testset "vi_normal" begin
        z = [1.0, 2.0]
        q = VariationalDistributions.vi_normal(z)
        @test q isa Bijectors.TransformedDistribution
        @test q.dist isa DistributionsAD.Normal
        @test q.dist.μ == z[1]
        @test q.dist.σ == LogExpFunctions.log1pexp(z[2])
    end

    # Test vi_constant_normal
    @testset "vi_constant_normal" begin
        z = [1.0]
        q = VariationalDistributions.vi_constant_normal(z)
        @test q isa Bijectors.TransformedDistribution
        @test q.dist isa ConstantDistribution
        @test q.dist.m == z[1]
    end

end

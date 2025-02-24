using Test
using Distributions
using Random

using MirrorVI: ConstantDistribution, MultivariateConstantDistribution


@testset "ConstantDistribution Tests" begin

    # Test Univariate ConstantDistribution
    @testset "Univariate ConstantDistribution" begin
        # Test constructor
        d = ConstantDistribution(5.0)
        @test d.m == 5.0

        # Test evaluation
        @test d(10.0) == 5.0
        @test d(-3.0) == 5.0

        # Test entropy
        @test entropy(d) == 0.0

        # Test sampling
        rng = MersenneTwister(42)  # Fixed RNG for reproducibility
        @test rand(rng, d) == 5.0
    end

    # Test Multivariate ConstantDistribution
    @testset "Multivariate ConstantDistribution" begin
        # Test constructor
        m = [1.0, 2.0, 3.0]
        d = MultivariateConstantDistribution(m)
        @test d.m == m

        # Test evaluation
        x = [0.5, 1.5, 2.5]
        @test d(x) == m

        # Test entropy
        @test entropy(d) == 0.0

        # Test sampling
        rng = MersenneTwister(42)  # Fixed RNG for reproducibility
        @test rand(rng, d) == m
    end

end

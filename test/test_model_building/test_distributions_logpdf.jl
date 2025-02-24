using Test
using LogExpFunctions: log1pexp
using SpecialFunctions: logbeta

using MirrorVI: DistributionsLogPdf


@testset "DistributionsLogPdf Module Tests" begin

    # Test log_normal with AbstractArray
    @testset "log_normal with AbstractArray" begin
        x = [1.0, 2.0, 3.0]
        mu = [0.0, 0.0, 0.0]
        sigma = [1.0, 1.0, 1.0]
        result = DistributionsLogPdf.log_normal(x, mu, sigma)
        expected_result = -0.5f0 * log.(2 * Float32(pi)) .- log.(sigma) .- 0.5f0 * ((x .- mu) ./ sigma).^2f0
        @test result ≈ expected_result
    end

    # Test log_normal with Real
    @testset "log_normal with Real" begin
        x = 1.0
        mu = 0.0
        sigma = 1.0
        result = DistributionsLogPdf.log_normal(x, mu, sigma)
        expected_result = -0.5f0 * log(2 * Float32(pi)) - log(sigma) - 0.5f0 * ((x - mu) / sigma)^2f0
        @test result ≈ expected_result
    end

    # Test log_half_normal with Real
    @testset "log_half_normal with Real" begin
        x = 1.0
        sigma = 1.0
        result = DistributionsLogPdf.log_half_normal(x, sigma)
        expected_result = 0.5f0 * log(2) - 0.5f0 * log(Float32(pi)) - log(sigma) - 0.5f0 * (x / sigma)^2f0
        @test result ≈ expected_result
    end

    # Test log_half_cauchy with AbstractArray
    @testset "log_half_cauchy with AbstractArray" begin
        x = [1.0, 2.0, 3.0]
        sigma = [1.0, 1.0, 1.0]
        result = DistributionsLogPdf.log_half_cauchy(x, sigma)
        expected_result = log(2f0) .- log.(Float32(pi) .* sigma) .- log.(1f0 .+ (x ./ sigma).^2f0)
        @test result ≈ expected_result
    end

    # Test log_half_cauchy with Real
    @testset "log_half_cauchy with Real" begin
        x = 1.0
        sigma = 1.0
        result = DistributionsLogPdf.log_half_cauchy(x, sigma)
        expected_result = log(2f0) - log(Float32(pi) * sigma) - log(1f0 + (x / sigma)^2f0)
        @test result ≈ expected_result
    end

    # Test log_beta with AbstractArray
    @testset "log_beta with AbstractArray" begin
        x = [0.1, 0.5, 0.9]
        a = 2.0
        b = 2.0
        result = DistributionsLogPdf.log_beta(x, a, b)
        expected_result = (a - 1) .* log.(x) .+ (b - 1) .* log.(1 .- x) .- logbeta(a, b)
        @test result ≈ expected_result
    end

    # Test log_bernoulli_from_logit with AbstractArray
    @testset "log_bernoulli_from_logit with AbstractArray" begin
        x = [0, 1, 0]
        logitp = [0.5, -0.5, 0.0]
        result = DistributionsLogPdf.log_bernoulli_from_logit(x, logitp)
        expected_result = @. - (1 - x) * log1pexp(logitp) - x * log1pexp(-logitp)
        @test result ≈ expected_result
    end

    # Test log_bernoulli_from_logit with Real
    @testset "log_bernoulli_from_logit with Real" begin
        x = 1
        logitp = 0.5
        result = DistributionsLogPdf.log_bernoulli_from_logit(x, logitp)
        expected_result = -log1pexp(-logitp)
        @test result ≈ expected_result
    end

    # Test log_bernoulli with AbstractArray
    @testset "log_bernoulli with AbstractArray" begin
        x = [0, 1, 0]
        prob = [0.1, 0.9, 0.5]
        result = DistributionsLogPdf.log_bernoulli(x, prob)
        expected_result = @. x * log(prob + DistributionsLogPdf.eps_prob) + (1. - x) * log(1. - prob + DistributionsLogPdf.eps_prob)
        @test result ≈ expected_result
    end

    # Test log_bernoulli with Real
    @testset "log_bernoulli with Real" begin
        x = 1
        prob = 0.9
        result = DistributionsLogPdf.log_bernoulli(x, prob)
        expected_result = x * log(prob + DistributionsLogPdf.eps_prob) + (1. - x) * log(1. - prob + DistributionsLogPdf.eps_prob)
        @test result ≈ expected_result
    end

end

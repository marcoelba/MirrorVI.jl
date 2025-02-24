using Test
using Bijectors
using LogExpFunctions

# Test Bijectors.jacobian for log1pexp with AbstractArray
@testset "Bijectors.jacobian for log1pexp with AbstractArray" begin
    x = [1.0, 2.0, 3.0]
    xt = log1pexp.(x)
    logdetjac = Bijectors.jacobian(log1pexp, x, xt)
    expected_logdetjac = sum(x .- xt)
    @test logdetjac ≈ expected_logdetjac
end

# Test Bijectors.jacobian for log1pexp with Real
@testset "Bijectors.jacobian for log1pexp with Real" begin
    x = 1.0
    xt = log1pexp(x)
    logdetjac = Bijectors.jacobian(log1pexp, x, xt)
    expected_logdetjac = x - xt
    @test logdetjac ≈ expected_logdetjac
end

# Test Bijectors.jacobian for identity with AbstractArray
@testset "Bijectors.jacobian for identity with AbstractArray" begin
    x = [1.0, 2.0, 3.0]
    xt = identity.(x)
    logdetjac = Bijectors.jacobian(identity, x, xt)
    expected_logdetjac = zero(eltype(x))
    @test logdetjac ≈ expected_logdetjac
end

# Test Bijectors.jacobian for identity with Real
@testset "Bijectors.jacobian for identity with Real" begin
    x = 1.0
    xt = identity(x)
    logdetjac = Bijectors.jacobian(identity, x, xt)
    expected_logdetjac = zero(eltype(x))
    @test logdetjac ≈ expected_logdetjac
end

# Test Bijectors.jacobian for logistic with AbstractArray
@testset "Bijectors.jacobian for logistic with AbstractArray" begin
    x = [1.0, 2.0, 3.0]
    xt = logistic.(x)
    logdetjac = Bijectors.jacobian(logistic, x, xt)
    expected_logdetjac = sum(log.(xt .* (1 .- xt)))
    @test logdetjac ≈ expected_logdetjac
end

# Test Bijectors.jacobian for logistic with Real
@testset "Bijectors.jacobian for logistic with Real" begin
    x = 1.0
    xt = logistic(x)
    logdetjac = Bijectors.jacobian(logistic, x, xt)
    expected_logdetjac = log(xt * (1 - xt))
    @test logdetjac ≈ expected_logdetjac
end

# Test LogExpFunctions.log1pexp with AbstractArray
@testset "LogExpFunctions.log1pexp with AbstractArray" begin
    x = [1.0, 2.0, 3.0]
    result = LogExpFunctions.log1pexp(x)
    expected_result = log1pexp.(x)
    @test result ≈ expected_result
end

# Test LogExpFunctions.logistic with AbstractArray
@testset "LogExpFunctions.logistic with AbstractArray" begin
    x = [1.0, 2.0, 3.0]
    result = LogExpFunctions.logistic(x)
    expected_result = logistic.(x)
    @test result ≈ expected_result
end

# test
using MirrorVI: Predictors
using Test
using ComponentArrays

# Test data
X = [1.0 2.0; 3.0 4.0]
theta = ComponentArray(beta0=1.0, beta=[2.0, 3.0], sigma_y=0.1)
theta_ri = ComponentArray(beta0_random=1.0, beta_fixed=[2.0, 3.0], sigma_y=0.1)
theta_time = ComponentArray(beta_time=[1.0, 2.0], beta_fixed=[2.0 3.0; 4.0 5.0], sigma_y=0.1)
theta_time_ri = ComponentArray(
    beta_time=[1.0 2.0; 3.0 4.0],
    beta_fixed=rand(2, 2, 2),
    sigma_beta=0.1,
    beta0_random=1.0,
    sigma_y=0.1
)

@testset "Predictors" begin

    @testset "linear_model" begin
        mu, sigma = Predictors.linear_model(theta, X=X)
        @test mu == [1.0 + 1.0*2.0 + 2.0*3.0, 1.0 + 3.0*2.0 + 4.0*3.0]
        @test sigma == [0.1, 0.1]
    end

    @testset "linear_predictor" begin
        mu = Predictors.linear_predictor(theta, X=X)[1]
        @test mu == [1.0 + 1.0*2.0 + 2.0*3.0, 1.0 + 3.0*2.0 + 4.0*3.0]
    end

    @testset "random_intercept_model" begin
        mu, sigma = Predictors.random_intercept_model(theta_ri, 1, X=X)
        @test mu == [1.0 + 1.0*2.0 + 2.0*3.0, 1.0 + 3.0*2.0 + 4.0*3.0]
        @test sigma == [0.1, 0.1]
    end

    @testset "linear_time_model" begin
        mu, sigma = Predictors.linear_time_model(theta_time, X=X)
        
        pred_t1 = X * theta_time[:beta_fixed][:, 1] .+ theta_time[:beta_time][1]
        pred_t2 = pred_t1 .+ X * theta_time[:beta_fixed][:, 2] .+ theta_time[:beta_time][2]
        @test mu == hcat(pred_t1, pred_t2)
        @test sigma == [0.1 0.1; 0.1 0.1]
    end

    @testset "linear_time_random_intercept_model" begin
        mu, sigma = Predictors.linear_time_random_intercept_model(theta_time_ri, 1, X=X)
        @test size(mu) == (2, 2)
        @test size(sigma) == (2, 2)
    end

end

using Test
using OrderedCollections
using ComponentArrays
using Distributions

using MirrorVI
using MirrorVI: polynomial_decay, cyclical_polynomial_decay, compute_logpdf_prior, get_parameters_axes, elbo, hybrid_training_loop, rand_z_dropout, Bijectors


# Test polynomial_decay
@testset "polynomial_decay" begin
    # Test default parameters
    @test polynomial_decay(10) ≈ 0.17782794f0 atol=1e-3
    @test polynomial_decay(100) ≈ 0.031234133f0 atol=1e-3

    # Test custom parameters
    @test polynomial_decay(10, a=2.0f0, b=0.1f0, gamma=0.5f0) ≈ 0.6294555f0 atol=1e-3
    @test polynomial_decay(100, a=0.5f0, b=0.01f0, gamma=1.0f0) ≈ 0.004950495f0 atol=1e-3
end

# Test cyclical_polynomial_decay
@testset "cyclical_polynomial_decay" begin
    # Test with default parameters
    schedule = cyclical_polynomial_decay(10, 2)
    @test length(schedule) == 10
    @test schedule[1] ≈ schedule[6] atol=1e-6

    # Test with custom parameters
    schedule = cyclical_polynomial_decay(12, 3)
    @test length(schedule) == 12
    @test schedule[1] ≈ schedule[5] atol=1e-6
    @test schedule[5] ≈ schedule[9] atol=1e-6
end

# Test compute_logpdf_prior
@testset "compute_logpdf_prior" begin
    # Create a mock params_dict
    params_dict = OrderedDict(
        "priors" => OrderedDict(
            "prior1" => Dict(
                "dependency" => [],
                "logpdf_prior" => (x, deps...) -> logpdf(Normal(0, 1), x)
            ),
            "prior2" => Dict(
                "dependency" => ["prior1"],
                "logpdf_prior" => (x, deps...) -> logpdf(Normal(deps[1], 1), x)
            )
        )
    )

    # Create a mock theta
    theta = ComponentArray(prior1=1.0, prior2=2.0)

    # Test compute_logpdf_prior
    @test compute_logpdf_prior(theta, params_dict=params_dict) ≈ logpdf(Normal(0, 1), 1.0) + logpdf(Normal(1.0, 1), 2.0)
end

# Test get_parameters_axes
@testset "get_parameters_axes" begin
    # Create a mock params_dict
    params_dict = OrderedDict(
        "priors" => OrderedDict(
            "prior1" => Dict("dim_theta" => (1,)),
            "prior2" => Dict("dim_theta" => (2,))
        )
    )

    # Test get_parameters_axes
    proto_array = ComponentArray(; (:prior1=>1.0, :prior2=>[1.0, 2.0])...)

    axes = get_parameters_axes(params_dict)
    @test axes == getaxes(proto_array)
end

# Test rand_z_dropout
@testset "rand_z_dropout" begin
    # Create a mock params_dict
    params_dict = OrderedDict(
        "priors" => OrderedDict(
            "prior1" => Dict("can_dropout" => true, "dim_z" => 4),
            "prior2" => Dict("can_dropout" => false, "dim_z" => 2)
        )
    )

    # Test rand_z_dropout
    z = rand_z_dropout(params_dict)
    @test length(z) == 6
    @test all(z[1:2] .∈ [0.0, 1.0])
    @test all(z[3:4] .== 1.0)
    @test all(z[5:6] .== 1.0)
end

# Test elbo (mock test, as it depends on external functions)
@testset "elbo" begin
    # Mock data and functions
    z = [1.0, 2.0]
    y = [0.0, 1.0]
    X = [1.0 2.0; 3.0 4.0]
    ranges_z = [1:1, 2:2]
    vi_family_array = [
        x -> Bijectors.transformed(Normal(x[1], 1.0), identity),
        x -> Bijectors.transformed(Normal(x[1], 1.0), identity)
    ]
    random_weights = [true, false]
    model = (theta; X) -> (theta[1] .* X[:, 1] .+ theta[2] .* X[:, 2], )

    proto_array = ComponentArray(; (:prior1=>1.0, :prior2=>1.0)...)
    theta_axes = getaxes(proto_array)
    log_likelihood = (y, pred) -> sum(logpdf.(Normal.(pred, 1.0), y))
    log_prior = theta -> 0.0

    # Test elbo
    res = elbo(z;
        y=y,
        X=X,
        ranges_z=ranges_z,
        vi_family_array=vi_family_array,
        random_weights=random_weights,
        model=model,
        theta_axes=theta_axes,
        log_likelihood=log_likelihood,
        log_prior=log_prior
    )
    @test res isa Float64
end

# Test hybrid_training_loop (mock test, as it depends on external functions)
@testset "hybrid_training_loop" begin
    # Mock data and functions
    z = [1.0, 2.0]
    y = [0.0, 1.0]
    X = [1.0 2.0; 3.0 4.0]
    params_dict = OrderedDict(
        "vi_family_array" => [
            x -> Bijectors.transformed(Normal(x[1], 1.0), identity),
            x -> Bijectors.transformed(Normal(x[1], 1.0), identity)
        ],
        "ranges_z" => [1:1, 2:2],
        "random_weights" => [true, false],
        "noisy_gradients" => [0, 0],
        "priors" => OrderedDict(
            "prior1" => Dict("dim_theta" => (1,)),
            "prior2" => Dict("dim_theta" => (1,))
    ))
    model = (theta; X) -> (theta[1] .* X[:, 1] .+ theta[2] .* X[:, 2], )
    log_likelihood = (y, pred) -> sum(logpdf.(Normal.(pred, 1.0), y))
    log_prior = theta -> 0.0
    optimiser = MirrorVI.Optimisers.Adam()

    # Test hybrid_training_loop
    result = hybrid_training_loop(
        z=z,
        y=y,
        X=X,
        params_dict=params_dict,
        model=model,
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        n_iter=10,
        optimiser=optimiser,
        use_noisy_grads=false
    )
    @test haskey(result, "loss_dict")
    @test haskey(result, "best_iter_dict")
end

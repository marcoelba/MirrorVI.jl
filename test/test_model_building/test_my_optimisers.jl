using Test
using Optimisers
using MirrorVI: MyOptimisers


ϵ = MyOptimisers.ϵ

@testset "MyOptimisers.DecayedADAGrad Tests" begin

    # Test initialization of DecayedADAGrad
    @testset "DecayedADAGrad Initialization" begin
        opt = MyOptimisers.DecayedADAGrad(η=0.1, pre=1.0, post=0.9)
        @test opt.eta == 0.1
        @test opt.pre == 1.0
        @test opt.post == 0.9
    end

    # Test state initialization
    @testset "State Initialization" begin
        opt = MyOptimisers.DecayedADAGrad(η=0.1, pre=1.0, post=0.9)
        x = [1.0, 2.0, 3.0]
        state = Optimisers.init(opt, x)
        @test state ≈ [ϵ, ϵ, ϵ]
    end

    # Test apply! method with a single update
    @testset "apply! Single Update" begin
        opt = MyOptimisers.DecayedADAGrad(η=0.1, pre=1.0, post=0.9)
        x = [1.0, 2.0, 3.0]
        dx = [0.1, 0.2, 0.3]
        state = Optimisers.init(opt, x)

        # Make a copy of the gradient to compare later
        original_dx = copy(dx)

        η = opt.eta
        acc = copy(state)
    
        expected_state = opt.post .* acc .+ opt.pre .* dx.^2
        expected_dx = dx .* η ./ (.√expected_state .+ ϵ)
    
        # Apply the optimiser
        new_state, updated_dx = Optimisers.apply!(opt, state, x, dx)
    
        # Check the updated state
        @test new_state ≈ expected_state
    
        # Check the updated gradient
        @test updated_dx ≈ expected_dx
    
        # Ensure the original dx is modified in place
        @test dx ≈ expected_dx
    end

    # Test apply! method with zero gradient
    @testset "apply! Zero Gradient" begin
        opt = MyOptimisers.DecayedADAGrad(η=0.1, pre=1.0, post=0.9)
        x = [1.0, 2.0, 3.0]
        dx = [0., 0., 0.]
        state = Optimisers.init(opt, x)

        # Make a copy of the gradient to compare later
        original_dx = copy(dx)

        η = opt.eta
        acc = copy(state)
    
        expected_state = opt.post .* acc .+ opt.pre .* dx.^2
        expected_dx = dx .* η ./ (.√expected_state .+ ϵ)

        # Apply the optimiser
        new_state, updated_dx = Optimisers.apply!(opt, state, x, dx)

        # Check the updated state
        @test new_state ≈ expected_state

        # Check the updated gradient
        @test updated_dx ≈ expected_dx
    end

end

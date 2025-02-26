module MyOptimisers

using Optimisers


const ϵ = 1e-8

"""
    DecayedADAGrad(; η = 0.1, pre = 1.0, post = 0.9)

Implements a decayed version of AdaGrad.
It has parameter specific learning rates based on how frequently it is updated.
Does not really need tuning.
References: [ADAGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimiser.

# Arguments
- `η=0.1`: learning rate
- `pre=1.0`: weight of new gradient norm
- `post=0.9`: weight of histroy of gradient norms

# Returns
- `Vector{Float32}`:

"""
struct DecayedADAGrad <: Optimisers.AbstractRule
    eta::Float64
    pre::Float64
    post::Float64
end

DecayedADAGrad(; η = 0.1, pre = 1.0, post = 0.9) = DecayedADAGrad(η, pre, post)

Optimisers.init(o::DecayedADAGrad, x::AbstractArray) = Optimisers.onevalue(ϵ, x)

function Optimisers.apply!(o::DecayedADAGrad, state, x::AbstractArray, dx)
    
    η = o.eta
    acc = state

    @. acc = o.post * acc + o.pre * dx^2
    @. dx *= η / (√acc + ϵ)

    return acc, dx
end

end

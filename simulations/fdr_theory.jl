# FDR theory
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using StatsFuns
using StatsBase
using HypothesisTests
using DataFrames


p0 = 45
p1 = 5
a0 = 0.01
a1 = 0.99
V = Distributions.Binomial(p0, a0)
T = Distributions.Binomial(p1, a1)

fdp = []
v_s = []
t_s = []
for sim =1:5000
    v = rand(V)
    append!(v_s, v)
    t = rand(T)
    append!(t_s, t)
    append!(fdp, v / (v + t))    
end
# FDR
mean(fdp)
mean(v_s .== 0) * mean(t_s .== 0)
(1 - a0)^p0

# marginal FDR
(p0 * a0) / (p0 * a0 + p1 * a1)


# Gaussian Mirror Lemma 1
U = Distributions.Normal(0, 0.5)
V = Distributions.Normal(0, 0.7)

UpV = Distributions.Normal(0, sqrt(U.σ^2 + V.σ^2))
UmV = Distributions.Normal(0, sqrt(U.σ^2 + V.σ^2))


p_m = abs.(rand(UpV, 5000)) - abs.(rand(UmV, 5000))
m_p = abs.(rand(UmV, 5000)) - abs.(rand(UpV, 5000))

StatsPlots.histogram(p_m)
StatsPlots.histogram!(m_p)

mean(p_m .> 1.)
mean(m_p .> 1.)
mean(m_p .< -1.)
mean(p_m .< -1.)


# Normal prior product
lambda_dist = Distributions.Beta(0.1, 0.1)
lambda = 0.5
tau = 1.
eta = Distributions.Normal(0, lambda * tau)
theta = eta * lambda
Distributions.Normal(0, tau) * lambda^2

# double expectation
p0 = 90
p1 = 10
p = p0 + p1
n = 1000

# data generating function
function data_gen(beta, n)
    X = rand(MvNormal(ones(p)), n)'
    y = X * beta + randn(n)*0.5
    return y
end

# Prior distribution
function beta_prior(a=1., b=1., t=0.1)
    beta = rand(Normal(0, 1), p)
    q = rand(Beta(a, b), p)
    zero_q = q .<= t
    beta[zero_q] .= 0.0
    return beta, zero_q
end

beta, zero_q = beta_prior()
y = data_gen(beta, n)


fdp = []
for sim = 1:1000

end

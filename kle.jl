using LinearAlgebra, Statistics

struct MC_KLE_1d
    lambda::Vector{Float64}
    psi::Matrix{Float64}
    mu::Vector{Float64}
end

## N_grid, N_MC = size(samples)
function form_KL_MC(samples::Matrix{Float64}, num_trunc::Int)
    mu = vec(mean(samples, dims=2))
    centered = samples .- mu
    C_mat = Hermitian(cov(centered, dims=2))
    lambda, phi = eigen(C_mat)
    psi = phi
    reverse!(lambda)
    reverse!(psi, dims=2)
    lambda = lambda[1:num_trunc]
    psi = psi[:,1:num_trunc]
    sigma = sqrt.(lambda)
    Z_samps = (psi' * centered) ./ sigma
    MC_KLE_1d(sigma, psi, mu), Z_samps
end
include("utils.jl")
include("kle.jl")
include("schloegl.jl")
using Catalyst, DifferentialEquations, CairoMakie, Random, Statistics, LinearAlgebra

cols = Makie.wong_colors()
save_plots = true
wait_for_key() = (print(stdout, "Press e to continue...\n"); read(stdin, 1); println(); nothing)

function KLE_create_msets(j::Int, basic_mset::MultiIndexSet, firstOrders::Vector{Int})
    f_d = length(firstOrders)+1
    basic_mset_mat = collect(basic_mset)
    j < f_d && return FixedMultiIndexSet(j, firstOrders[j])
    num_zero_cols = j-f_d
    mset_mat = [basic_mset_mat[:,1:f_d-1] zeros(Int, size(basic_mset_mat,1), num_zero_cols) basic_mset_mat[:,f_d]]
    Fix(MultiIndexSet(mset_mat))
end

function KLE_multi_index_setup(inputDim, outputDim, minOrder, firstOrders)
    f_d = length(firstOrders)+1
    basic_mset = CreateTotalOrder(f_d, minOrder)
    start_d = inputDim-outputDim+1
    msets = map(j->KLE_create_msets(j, basic_mset, firstOrders), start_d:inputDim)
    opts = MapOptions(basisType="HermiteFunctions", basisLB=-3, basisUB=3)
    msets, opts
end

function PlotRealizations(samples::Matrix{Float64}; tgrid=0.1:0.1:20., num_traj = 100, save_plot=save_plots, scale=1e-2, opacity=0.35)
    fig_real = Figure(backgroundcolor=(:white,0))
    ax_real = Axis(fig_real[1,1], backgroundcolor=(:white,0))
    for j in 1:min(num_traj, size(samples,2))
        lines!(ax_real, tgrid, samples[:,j] * scale, color = (cols[mod1(j, length(cols))], opacity))
    end
    hidedecorations!(ax_real, label=false, ticks=false, ticklabels=false)
    hidespines!(ax_real, :t,:r)

    save_plot && save("figs/realizations.pdf", fig_real)
    display(fig_real)
    fig_real
end

function TrainTotalOrderMap(samples, maxOrder, verbose, basisType="HermiteFunctions", basisLB=-3, basisUB=3)
    N_dim = size(samples,1)
    mapOpts = MapOptions(;basisType, basisLB, basisUB)
    trimap = CreateTriangular(N_dim, N_dim, maxOrder, mapOpts)
    obj = CreateGaussianKLObjective(samples)
    train_opts = TrainOptions(;verbose)
    train_time = @elapsed TrainMap(trimap, obj, train_opts)
    verbose && println("Dim $N_dim, Order $maxOrder, Training time: $train_time s")
    trimap
end

function Generate2dPullbackPlot(samples, rng, verbose, N_train=nothing, maxOrder=10, N_pullback=5000, opacity=0.25, save_plot=save_plots)
    isnothing(N_train) && (N_train = size(samples,2) ÷ 2)
    z1 = samples[1:2, 1:N_train]
    z2 = samples[1:2, N_train+1:end]
    trimap = TrainTotalOrderMap(z1, maxOrder, verbose)
    test_samples = randn(rng, 2, N_pullback)
    # Need an empty prefix for a square map inversion
    prefix = zeros(0, N_pullback)
    pullback_samps = Inverse(trimap, prefix, test_samples)

    fig = Figure(backgroundcolor=(:white,0))
    ax = Axis(fig[1,1], backgroundcolor=(:white,0))
    scatter!(ax, pullback_samps, color=(cols[1], opacity), label="Generated")
    scatter!(ax, z2, color=(cols[2], opacity), label="Out-of-sample")
    hidedecorations!(ax, label=false, ticks=false, ticklabels=false)
    hidespines!(ax, :t,:r)
    axislegend(ax, position=:rb, backgroundcolor=(:white,0))
    display(fig)
    save_plot && save("figs/pullback_2d.pdf", fig)

    trimap, fig
end

function PullbackScattermat(trimap, rng, verbose, save_plot=save_plots, N_test=1000, N_plot_modes=8)
    test_samples = randn(rng, inputDim(trimap), N_test)
    prefix = zeros(0, N_test)
    pullback_samps = Inverse(trimap, prefix, test_samples)
    fig = scattermat(pullback_samps[1:N_plot_modes,:])
    display(fig)
    verbose && wait_for_key()
    save_plot && save("figs/pullback_scattermat.pdf", fig)
    fig
end

function train_KL_map(samples, minOrder, firstOrders, verbose)
    num_KL_modes = size(samples,1)
    multiindex_sets, opts = KLE_multi_index_setup(num_KL_modes, num_KL_modes, minOrder, firstOrders)
    map_comps = [CreateComponent(mset, opts) for mset in multiindex_sets]
    trimap = TriangularMap(map_comps)
    obj = CreateGaussianKLObjective(samples)
    train_opts = TrainOptions(;verbose)
    train_time = @elapsed TrainMap(trimap, obj, train_opts)
    verbose && println("Marginal Training time: $train_time s, Dim $num_KL_modes, Coeffs $(numCoeffs(trimap))")
    trimap
end

# Creates coupling between trajectory at time t_cond and the sample of KLE stochastic params
function train_cond_KL_map(rng, Z_samples, traj_samples, minOrder, firstOrders, t_cond, verbose, noise=1e-1)
    traj_cond = traj_samples[t_cond,:] + noise*randn(rng, size(traj_samples,2))
    num_KL_modes = size(Z_samples,1)
    mu_cond = mean(traj_cond)
    std_cond = std(traj_cond)
    traj_cond .-= mu_cond
    traj_cond ./= std_cond
    samples = [traj_cond'; Z_samples]

    multiindex_sets, opts = KLE_multi_index_setup(num_KL_modes+1, num_KL_modes, minOrder, firstOrders)
    map_comps = [CreateComponent(mset, opts) for mset in multiindex_sets]
    trimap = TriangularMap(map_comps)
    obj = CreateGaussianKLObjective(samples, num_KL_modes)
    train_opts = TrainOptions(verbose=verbose)
    train_time = @elapsed TrainMap(trimap, obj, train_opts)
    verbose && println("Conditional Training time: $train_time s, Dim $(num_KL_modes+1), Coeffs $(numCoeffs(trimap))")
    trimap, (mu_cond, std_cond)
end

function Plot2dConditionalPullback(Z, pullback_samps, Z_real, comps=(1,2), save_plot=save_plots)
    fig = Figure(backgroundcolor=(:white,0))
    ax = Axis(fig[1,1],backgroundcolor=(:white,0))
    scatter!(ax, Z[comps[1],:], Z[comps[2],:], color=(cols[2], 0.4), label="Training Samples")
    scatter!(ax, pullback_samps[comps[1],:], pullback_samps[comps[2],:], color=(cols[1], 0.4), label="Pullback")
    scatter!(ax, Z_real[comps[1]], Z_real[comps[2]], color=:black, label="Conditioned Sample")
    axislegend(ax, position=:rb)
    hidedecorations!(ax)
    hidespines!(ax)
    display(fig)
    save_plot && save("figs/conditional_pullback_2d.pdf", fig)
    fig
end

function PlotConditionalPullback(rngs, schloegl, kle, Z_samples, trimap, sample_transform, t_cond, verbose,
        tgrid=0.1:0.1:20, N_test=100, save_plot=save_plots)
    new_traj = SampleSchloegl(schloegl, rngs; n_samples=1)
    num_KL_modes = size(Z_samples,1)
    kle_proj = (kle.psi' * (new_traj .- kle.mu) ./ kle.lambda)'
    mu_cond, std_cond = sample_transform

    test_samples = randn(rngs[1], num_KL_modes, N_test)
    prefix = fill((new_traj[t_cond] - mu_cond)/std_cond, 1, N_test)
    pullback_samps = Inverse(trimap, prefix, test_samples)
    fig_2d = Plot2dConditionalPullback(Z_samples, pullback_samps, kle_proj)
    verbose && wait_for_key()

    traj_cond_samps = kle.psi * (kle.lambda .* pullback_samps) .+ kle.mu

    fig_cond_traj = Figure(backgroundcolor=(:white,0))
    ax = Axis(fig_cond_traj[1,1],backgroundcolor=(:white,0))
    for j in axes(traj_cond_samps,2)
        lines!(ax, tgrid, traj_cond_samps[:,j]./100, color = (cols[mod1(j, length(cols))], 0.35))
    end
    hlines!(ax, new_traj[t_cond]/100, color=:black)
    vlines!(ax, tgrid[t_cond], color=:black)
    lines!(ax, tgrid, new_traj[:]./100, color=:black)
    hidedecorations!(ax, label=false, ticks=false, ticklabels=false)
    hidespines!(ax, :t,:r)
    display(fig_cond_traj)
    save_plot && save("figs/conditional_realizations.pdf", fig_cond_traj)
    fig_2d, fig_cond_traj
end

function reaction_experiment(verbose=true)
    rngs = [Xoshiro(284022 + 28403*j) for j in 1:Threads.nthreads()]
    schloegl = SchloeglNetwork()
    n_samples, n_KL_modes = 2000, 30
    traj_samples = SampleSchloegl(schloegl, rngs; n_samples)
    PlotRealizations(traj_samples)
    verbose && wait_for_key()
    kle, Z_samples = form_KL_MC(traj_samples, n_KL_modes)

    samples_scatter = scattermat(Z_samples[1:8,:])
    display(samples_scatter)
    save_plots && save("figs/samples_scatter.pdf", samples_scatter)
    verbose && wait_for_key()

    Generate2dPullbackPlot(Z_samples, rngs[1], verbose)
    verbose && wait_for_key()

    minOrder, firstOrders = 4, [15, 10]
    marginal_kl_map = train_KL_map(Z_samples, minOrder, firstOrders, verbose)
    PullbackScattermat(marginal_kl_map, rngs[1], verbose)

    t_cond = 20
    conditional_kl_map, sample_transform = train_cond_KL_map(rngs[1], Z_samples, traj_samples, minOrder, firstOrders, t_cond, verbose)
    PlotConditionalPullback(rngs, schloegl, kle, Z_samples, conditional_kl_map, sample_transform, t_cond, verbose)
end

##
reaction_experiment()
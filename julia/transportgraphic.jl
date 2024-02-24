using CairoMakie, Distributions, Random, LinearAlgebra, JLD2, MParT, ColorSchemes
using ColorSchemes: RGBA
cols = Makie.wong_colors()
save_plots = false

# Generate a random GMM distribution according to heuristics
# This is just for a generic target distribution--
# any (reasonable) 2d target here will work
function generate_gmm(rng, num_modes)
    flip = [1 0;0 -1]
    center_theta = map(j->2π*rand(rng)/num_modes + (j-1)*2π/num_modes, 1:num_modes)
    center_rad = map(_->2exp(randn(rng)), 1:num_modes)
    centers = [r*[cos(t), sin(t)] for (r,t) in zip(center_rad, center_theta)]
    cov_eig = map(_->0.1 .+ 1.5exp.(randn(rng,2)), 1:num_modes)
    cov_Q = map(_->collect(qr(randn(rng, 2)).Q), 1:num_modes)
    covs = [Hermitian(Q*Diagonal(eig)*Q') for (Q,eig) in zip(cov_Q, cov_eig)]
    dists = [MvNormal(flip*μ,flip*Σ*flip) for (μ,Σ) in zip(centers, covs)]
    weights = 0.5 .+ 2rand(rng,num_modes)
    weights /= sum(weights)
    MixtureModel(dists, weights)
end

# Create a figure of the target distribution PDF
function example_target_fig(gmm, grid_pts, save_plot = save_plots)
    gmm_pdf = [pdf(gmm, [x,y]) for x in grid_pts, y in grid_pts]
    fig = Figure(backgroundcolor = (:white,0))
    ax = Axis(fig[1,1], backgroundcolor = (:white,0))
    ax.aspect = 1
    contour!(ax, grid_pts, grid_pts, gmm_pdf, linewidth=3)
    hidedecorations!(ax)
    hidespines!(ax)
    save_plot && save("../figs/example_target.pdf", fig)
    fig, gmm_pdf
end

# Create a figure of the reference distribution PDF
function example_reference_fig(grid_pts, save_plot=save_plots)
    z = [exp(-(x^2+y^2)/2) for x in grid_pts, y in grid_pts]
    fig = Figure(backgroundcolor = (:white,0))
    ax = Axis(fig[1,1], backgroundcolor = (:white,0))
    ax.aspect = 1
    contour!(ax, grid_pts, grid_pts, z, linewidth=3)
    hidedecorations!(ax)
    hidespines!(ax)
    save_plot && save("../figs/example_reference.pdf", fig)
    fig
end

# Train a map on the target distribution
function example_create_map_gmm(rng, gmm, totalOrder = 3, N_samples = 10_000)
    samples = rand(rng, gmm, N_samples)
    # Normalize samples (i.e. match N(0,1) distribution in marginals)
    μ,σ = mean(samples, dims=2), std(samples, dims=2)
    samples .-= μ
    samples ./= σ

    # Create map
    map_dim = length(gmm)
    mapOpts = MapOptions(basisType="HermiteFunctions", basisLB=-4, basisUB=4)
    trimap = CreateTriangular(map_dim, map_dim, totalOrder, mapOpts)

    # Train map example
    obj = CreateGaussianKLObjective(samples)
    train_opts = TrainOptions()
    TrainMap(trimap, obj, train_opts)
    trimap, μ, σ
end

# Sample a trained map to create pullback samples
function create_pullback_samples(rng, trimap, μ, σ, N_test = 5000)
    test_samples = randn(rng, inputDim(trimap), N_test)
    prefix = zeros(0, N_test)
    pullback_samps = (σ .* Inverse(trimap, prefix, test_samples)) .+ μ
    test_samples, pullback_samps
end

function example_before_after_transport(rng, trimap, μ, σ; N_test=3000, save_plot=save_plots)
    test_samps, pullback_samps = create_pullback_samples(rng, trimap, μ, σ, N_test)

    # boilerplate code to get colors for points (via distance from origin in reference)
    radii = sum(abs.(test_samps).^2, dims=1).^(0.25)
    color_nums = (radii ./ maximum(radii))
    pt_cols = [RGBA(get(ColorSchemes.thermal, c),0.4) for c in color_nums[:]]
    # Format the figure as before and after
    fig = Figure(size=(1000,500), backgroundcolor = (:white,0))
    gl = fig[1,1] = GridLayout()
    ax = Axis(gl[1,1], backgroundcolor = (:white,0))
    ax2 = Axis(gl[1,2], backgroundcolor = (:white,0))
    colgap!(gl, -100)
    ax.aspect = 1
    ax2.aspect = 1
    xlims!(ax, (-4,6))
    ylims!(ax, (-5,5))
    xlims!(ax2, (-4,6))
    ylims!(ax2, (-5,5))

    # Plot the before and after with the contours of target
    contour!(ax, grid_pts, grid_pts, z, linewidth=3)
    scatter!(ax, test_samps[1,:], test_samps[2,:], markersize=10, color=pt_cols)
    contour!(ax2, grid_pts, grid_pts, z, linewidth=3)
    scatter!(ax2, pullback_samps[1,:], pullback_samps[2,:], markersize=10, color=pt_cols)

    # format the figs
    hidedecorations!(ax)
    hidespines!(ax)
    hidedecorations!(ax2)
    hidespines!(ax2)
    save_plot && save("../figs/example_pullback_before_after.pdf", fig)
    fig
end

# A simple animation of the action of the pullback
function animate_pullback(test_samples, pullback_samps, grid_pts, gmm_pdf)
    frame = Observable(1)
    xs = @lift(pullback_samps[1,:]*($frame-1)/num_frames + test_samples[1,:]*(num_frames-$frame)/num_frames)
    ys = @lift(pullback_samps[2,:]*($frame-1)/num_frames + test_samples[2,:]*(num_frames-$frame)/num_frames)

    fig = Figure()
    ax = Axis(fig[1,1])
    ax.aspect = 1
    contour!(ax, grid_pts, grid_pts, gmm_pdf, linewidth=3)
    scatter!(ax, xs, ys, markersize=7, color=(cols[3], 0.3))
    hidedecorations!(ax)
    hidespines!(ax)
    record(fig, "../figs/example_pullback.mp4", 1:num_frames; framerate=45) do j
        frame[] = j
    end
end

function example_conditional(gmm, grid_pts, cond_grid=-6:0.05:10, conditional_pt=-0.5, save_plot=save_plots)
    fig = Figure(size=(1000,500), backgroundcolor = (:white,0))
    ax = Axis(fig[1,1], backgroundcolor = (:white,0))
    ax2 = Axis(fig[1,2], backgroundcolor = (:white,0))
    ax.aspect = 1
    contour!(ax, grid_pts, grid_pts, z, linewidth=3)
    vlines!(ax, conditional_pt, color=:black, linestyle=:dash, linewidth=3)
    hidedecorations!(ax)
    hidespines!(ax)
    pdf_eval = pdf(gmm, [conditional_pt*ones(length(cond_grid))'; cond_grid'])
    band!(ax2, cond_grid, 0., pdf_eval, color=(cols[1],0.7))
    hidedecorations!(ax2)
    hidespines!(ax2)
    save_plot && save("../figs/example_conditional.pdf", fig)
    fig
end

rng = Xoshiro(2142)

gmm = generate_gmm(rng, 4)
@save "../data/gmm.jld2" gmm
grid_pts = -5:0.05:5
fig, z = example_target_fig(gmm, grid_pts)
trimap,μ,σ = example_create_map_gmm(rng, gmm)

##
# Create a before and after of pullback
# Color on distance from origin in reference
fig = example_before_after_transport(rng, trimap, μ, σ, save_plot=true)

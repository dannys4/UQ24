using MParT, BenchmarkTools, CairoMakie, Random, SpecialFunctions, CxxWrap, ProgressMeter, JLD2, Colors
# Define the inverse of the normal cumulative distribution function
invnormcdf(q) = sqrt(2)*erfinv(2*q-1)

# Default centers for one given level of sigmoids
default_centers = j-> invnormcdf.((1:j) ./ (j+1))
# Create a vector of default centers for a sigmoid component
create_default_centers = TO-> [-3; 3; reduce(vcat, default_centers.(1:TO))]

# Module to hold enums s.t. we are able to use them in the fix function
module TimingEnums
    @enum MapType MVE=1 RMVE=2 Integrand=3
    @enum EvalType Evaluate=1 Inverse=2 Gradient=3
    MapTypeToString = string.(instances(MapType))
    EvalTypeToString = string.(instances(EvalType))
end

# Define types to evaluate the correct function for fixed samples
abstract type FunctionFix end
struct EvalFix <: FunctionFix
    samples::Matrix{Float64}
end

struct InverseFix <: FunctionFix
    samples::Matrix{Float64}
    prefix::Matrix{Float64}
end

struct GradientFix <: FunctionFix
    samples::Matrix{Float64}
    sense::Matrix{Float64}
end

# Define the fix function to evaluate the correct function for fixed samples
function fix(f::TimingEnums.EvalType, samples::Matrix{Float64})
    if f == TimingEnums.Evaluate
        return EvalFix(samples)
    elseif f == TimingEnums.Inverse
        return InverseFix(samples, zeros(size(samples,1)-1, size(samples,2)))
    elseif f == TimingEnums.Gradient
        return GradientFix(samples, ones(1, size(samples,2)))
    end
end

# How to evaluate each object on a given map
function (e::EvalFix)(f)
    Evaluate(f, e.samples)
end

function (e::InverseFix)(f)
    Inverse(f, e.prefix, e.samples)
end

function (e::GradientFix)(f)
    Gradient(f, e.samples, e.sense)
end

function CreateFixedMsets(in_dim, max_order)
    total_fmset = MParT.CreateMarginalOrder(in_dim, max_order, -1)
    off_fmset = MParT.CreateMarginalOrder(in_dim-1, max_order, -1)
    diag_fmset = MParT.CreateMarginalOrder(in_dim, max_order, in_dim-1)
    (off_fmset, diag_fmset, total_fmset)
end

# Create the types of maps we're interested in
function CreateMaps(in_dim, max_order, opts)
    OUT_DIM = 1 # We're only interested in scalar outputs

    off_fmset, diag_fmset, total_fmset = CreateFixedMsets(in_dim, max_order)

    comp = CreateComponent(total_fmset, opts)
    mve = MParT.CreateExpansion(OUT_DIM, total_fmset, opts)

    ## Create sigmoid component
    centers = create_default_centers(max_order)
    sig_comp = MParT.CreateSigmoidComponent(off_fmset, diag_fmset, centers, opts)

    ## Create maps vector
    all_maps = Vector(undef, length(instances(TimingEnums.MapType)))
    all_maps[Int(TimingEnums.MVE)] = mve
    all_maps[Int(TimingEnums.RMVE)] = sig_comp
    all_maps[Int(TimingEnums.Integrand)] = comp
    all_maps
end

function CreateBenchmarks(rng, in_dim, sample_grid, all_maps)

    benchmark_size = (length(instances(TimingEnums.MapType)), length(instances(TimingEnums.EvalType)), length(sample_grid))
    benchmarks = Array{Float64,3}(undef, benchmark_size...)
    numIter = sum(sample_grid)*length(all_maps)*length(instances(TimingEnums.EvalType))
    numIter -= sum(sample_grid) # We don't want to benchmark the Inverse function for the MVE map
    prog = Progress(numIter, 1, "Benchmarking...")

    for (i, map_bench) in enumerate(all_maps)
        for (j, eval) in enumerate(instances(TimingEnums.EvalType))
            if eval == TimingEnums.Inverse && !isa(map_bench, CxxWrap.StdLib.SharedPtr{<:MParT.ConditionalMapBase})
                continue
            end
            for (k, N) in enumerate(sample_grid)
                f = fix(eval, randn(rng, in_dim, N))
                benchmarks[i,j,k] = @belapsed ($f)($map_bench)
                next!(prog, step=N)
            end
        end
    end
    benchmarks
end

function PlotBenchmarks(sample_grid, benchmarks; cols = Makie.wong_colors(), plot_filename=nothing)
    fig = Figure(backgroundcolor=(:white,0))
    markers = [:circle, :xcross, :diamond]
    markersizes = [13, 20, 13]
    linestyles = [:solid, :dash, :dot]
    xticks = 10 .^ (2:4)
    ax = Axis(fig[1,1], backgroundcolor=(:white,0), xticks=xticks, xlabel="Number of samples", ylabel="Time (s)", xscale=log10, yscale=log10)
    line_elems = [LineElement(color=cols[i], linestyle=linestyles[i], linewidth=3) for i in eachindex(TimingEnums.MapTypeToString)]
    scatter_elems = [MarkerElement(marker=markers[j], color=:black, markersize=markersizes[j]) for j in eachindex(TimingEnums.EvalTypeToString)]
    all_elems = [line_elems, scatter_elems]
    all_labels = [[TimingEnums.MapTypeToString...], [TimingEnums.EvalTypeToString...]]
    ylims!(ax, (1e-4, 2e2))
    for idx in CartesianIndices(benchmarks[:,:,1])
        i, j = Tuple(idx)
        i == Int(TimingEnums.MVE) && j == Int(TimingEnums.Inverse) && continue
        color = cols[i]
        linewidth = 3
        marker = markers[j]
        markersize=markersizes[j]
        linestyle=linestyles[i]
        if i == Int(TimingEnums.RMVE) && j == Int(TimingEnums.Inverse)
            color *= 0.6
        end
        scatterlines!(ax, sample_grid, benchmarks[i,j,:];
            marker, markersize, linestyle, color, linewidth)
    end
    leg = Legend(
        fig[1, 1], all_elems, all_labels, ["Map", "Function"],
        tellheight = false,
        tellwidth = false,
        orientation=:horizontal,
        halign = :right, valign = :bottom,
        backgroundcolor=(:white,0)
    )
    leg.nbanks=3
    leg.titleposition=:top
    isnothing(plot_filename) || save(plot_filename, fig)
    fig
end

function example_sample_benchmark(in_dim=1000, max_order=10, sample_grid = round.(Int, 10 .^ (2:0.33:4)))
    rng = Xoshiro(2048202)
    opts = MapOptions(basisType="HermiteFunctions", basisLB=-3, basisUB=3)
    all_maps = CreateMaps(in_dim, max_order, opts)
    benchmarks = CreateBenchmarks(rng, in_dim, sample_grid, all_maps)
    benchmark_keys = [(TimingEnums.MapTypeToString[idx[1]],
                       TimingEnums.EvalTypeToString[idx[2]],
                       sample_grid[idx[3]]) for idx in CartesianIndices(benchmarks)]
    benchmarks_dict = Dict(key=>val for (key, val) in zip(benchmark_keys, benchmarks))
    map_num_coeffs = numCoeffs.(all_maps)
    @save "../data/sample_benchmarks2.jld2" sample_grid benchmarks benchmarks_dict map_num_coeffs
end

@load "../data/sample_benchmarks2.jld2" sample_grid benchmarks benchmarks_dict map_num_coeffs
PlotBenchmarks(sample_grid, benchmarks, plot_filename="../figs/benchmarks.pdf")

##
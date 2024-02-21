using MParT, BenchmarkTools, CairoMakie, Random, SpecialFunctions, CxxWrap, ProgressMeter, JLD2
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

# Create the types of maps we're interested in
function CreateMaps(in_dim, out_dim, total_order, opts)
    fmset = FixedMultiIndexSet(in_dim, total_order)
    comp = CreateComponent(fmset, opts)
    mve = MParT.CreateExpansion(out_dim, fmset, opts)

    ## Create sigmoid component
    fmset_off = FixedMultiIndexSet(in_dim-1, total_order)
    fmset_diag = Fix(MParT.CreateNonzeroDiagTotalOrder(in_dim, total_order))
    centers = create_default_centers(total_order)
    sig_comp = MParT.CreateSigmoidComponent(fmset_off, fmset_diag, centers, opts)

    ## Create maps vector
    all_maps = Vector(undef, length(instances(TimingEnums.MapType)))
    all_maps[Int(TimingEnums.MVE)] = mve
    all_maps[Int(TimingEnums.RMVE)] = sig_comp
    all_maps[Int(TimingEnums.Integrand)] = comp
    all_maps
end

function CreateBenchmarks(rng, in_dim, sample_grid, all_maps)

    benchmarks = Array{Float64,3}(undef, length(instances(TimingEnums.MapType)), length(instances(TimingEnums.EvalType)), length(sample_grid))

    prog = Progress(sum(sample_grid)*length(all_maps)*length(instances(TimingEnums.EvalType)), 1, "Benchmarking...")

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
    fig = Figure()
    markers = [:circle, :xcross, :diamond]
    linestyles = [:solid, :dash, :dot]
    ax = Axis(fig[1,1], xlabel="Number of samples", ylabel="Time (s)", xscale=log10, yscale=log10)
    line_elems = [LineElement(color=cols[i], linestyle=linestyles[i], linewidth=3) for i in 1:length(instances(TimingEnums.MapType))]
    scatter_elems = [MarkerElement(marker=markers[j], color=:black, markersize=13) for j in 1:length(instances(TimingEnums.EvalType))]
    all_elems = [line_elems; scatter_elems]
    all_labels = [string.(instances(TimingEnums.MapType))..., string.(instances(TimingEnums.EvalType))...]

    for idx in CartesianIndices(benchmarks[:,:,1])
        i, j = Tuple(idx)
        i == Int(TimingEnums.MVE) && j == Int(TimingEnums.Inverse) && continue
        scatterlines!(ax, sample_grid, benchmarks[i,j,:], marker=markers[j], color=cols[i], linestyle=linestyles[i], linewidth=3, markersize=13)
    end
    Legend(
        fig[1, 1], all_elems, all_labels,
        tellheight = false,
        tellwidth = false,
        margin = (10, 10, 10, 10),
        halign = :right, valign = :bottom, orientation = :vertical
    )
    isnothing(plot_filename) || save(plot_filename, fig)
    fig
end

function example_sample_benchmark()
    in_dim = 10
    out_dim = 1
    total_order = 4
    sample_grid = round.(Int, 10 .^ (2:0.33:4.5))
    rng = Xoshiro(2048202)
    opts = MapOptions(basisType="HermiteFunctions", basisLB=-3, basisUB=3)
    all_maps = CreateMaps(in_dim, out_dim, total_order, opts)
    benchmarks = CreateBenchmarks(rng, in_dim, sample_grid, all_maps)
    benchmarks_dict = Dict((string(instances(TimingEnums.MapType)[idx[1]]), string(instances(TimingEnums.EvalType)[idx[2]]), sample_grid[idx[3]])=>(benchmarks[idx]) for idx in CartesianIndices(benchmarks))
    @save "data/sample_benchmarks.jld2" sample_grid benchmarks benchmarks_dict
end

##
@load "data/sample_benchmarks.jld2" sample_grid benchmarks benchmarks_dict
PlotBenchmarks(sample_grid, benchmarks, plot_filename="figs/benchmarks.pdf")

##
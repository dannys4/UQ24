import Base: show
using MParT, JLD2
"""
    scattermat(dataset;markersize=2,opacity=0.5,color=0,resolution=(700,700), gridlayout=nothing, row_offset=0, col_offset=0, kwargs...)
Scatterplot matrix of dataset, where each row is a variable and each column is a sample.
"""
function scattermat(dataset;markersize=2,opacity=0.5,color=:red,bins=50,gridlayout=nothing, row_offset=0, col_offset=0, kwargs...)
    n = size(dataset, 1)
    fig = nothing
    gg = gridlayout
    if isnothing(gg)
        fig = Figure()
        gg = fig[1,1] = GridLayout()
    end
    absmax = 1.1*maximum(abs.(dataset))
    for i in 1:n
        for j in i:n
            ax = Axis(gg[i+row_offset,j+col_offset])
            ax.aspect=1.
            if i == j
                hist!(ax, dataset[i,:], color=color, bins=bins; kwargs...)
                xlims!(ax, -absmax, absmax)
            else
                scatter!(ax, dataset[j,:], dataset[i,:], color=(color,opacity); markersize, kwargs...)
                xlims!(ax, -absmax, absmax)
                ylims!(ax, -absmax, absmax)
            end
            hidedecorations!(ax)
        end
    end
    colgap!(gg, 5)
    rowgap!(gg, 5)
    isnothing(fig) ? gridlayout : fig
end

function Base.show(io::IO, m::MultiIndex)
    print(io, string(m))
end

function Base.show(io::IO, m::MultiIndexSet)
    for j in 1:Size(m)
        println(io, m[j])
    end
end

function Base.collect(mset::MultiIndexSet)
    reduce(vcat, Int.(vec(mset[j]))' for j in 1:Size(mset))
end
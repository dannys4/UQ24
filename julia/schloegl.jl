using Catalyst, DifferentialEquations, Random
function SchloeglNetwork(k_1=3e-7, k_2=1e-4, k_3=1e-3, k_4=3.5, A=1e5, B=2e5)
    schloegl = @reaction_network begin
        @species X(t)
        @parameters _k_1=$k_1 _k_2=$k_2 _k_3=$k_3 _k_4=$k_4 _A=$A [isconstantspecies=true] _B=$B [isconstantspecies=true]
        (_k_1,_k_2), _A+2*X <--> 3*X
        (_k_3,_k_4), _B <--> X
    end
    schloegl
end

function SampleSchloegl(schloegl, rngs; X_0=250, n_samples=2000, t_end=20., t_step=0.1)
    @assert length(rngs) == Threads.nthreads() "Number of rngs must match number of threads"
    dprob = DiscreteProblem(schloegl, [:X=>X_0], (0., t_end))
    jprobs = [JumpProblem(schloegl, dprob, Direct();rng) for rng in rngs]
    tgrid = t_step:t_step:t_end
    X = Matrix{Float64}(undef, length(tgrid), n_samples)
    Threads.@threads for j in axes(X,2)
        id = Threads.threadid()
        sol = solve(jprobs[id], SSAStepper())
        X[:,j] .= vec(sol(tgrid))
    end
    X
end

function SampleSchloeglExample(n_samples)
    rngs = [Xoshiro(284022 + 28403*j) for j in 1:Threads.nthreads()]
    schloegl = SchloeglNetwork()
    SampleSchloegl(schloegl, rngs, n_samples=n_samples)
end
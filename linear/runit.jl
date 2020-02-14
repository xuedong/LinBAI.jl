using Random;
using CPUTime;
using LinearAlgebra;
include("pep_linear.jl");
include("../expfam.jl");
include("samplingrules.jl");

# Run the learning algorithm, paramterised by a sampling rule
# The stopping and recommendation rules are common
#
# βs must be a list of thresholds *in increasing order*

function play!(i, k, rng, pep, µ, S, N, P, Vinv)
    arm = pep.arms[k]
    Y = sample(rng, getexpfam(pep, 1), arm'µ)
    S .+= Y .* arm
    Vinv .= sherman_morrison(Vinv, arm)
    N[k] += 1
end

function runit(seed, sr, μs, pep::Union{LinearBestArm,LinearThreshold}, βs, δs)
    # seed: random seed. UInt.
    # sr: sampling rule.
    # µs: mean vector.
    # pep: pure exploration problem.
    # βs: list of thresholds.
    convex_sr = typeof(sr) == ConvexGame  # test if P is needed.

    βs = collect(βs) # mutable copy

    rng = MersenneTwister(seed)

    K = narms(pep, µs)
    nb_I = nanswers(pep, µs)
    dim = length(μs)
    convex_sr ? P = ones(Int64, (nb_I, K)) : P = ones(Int64, (1, 1))  # counts detailed by answer
    N = zeros(Int64, K)              # counts
    S = zeros(dim)                     # sum of samples
    Vinv = Matrix{Float64}(I, dim, dim)  # inverse of the design matrix

    baseline = CPUtime_us()

    # pull each arm once. TODO: pull instead d arms spanning the space.
    for k = 1:K
        play!(1, k, rng, pep, µs, S, N, P, Vinv)
    end

    state = start(sr, N, P)

    R = Tuple{Int64,Array{Int64,1},Array{Int64,2},UInt64}[] # collect return values

    while true
        t = sum(N)

        hµ = Vinv * S  # emp. estimates
        #println("hµ : $(round.(hµ, digits=3)); Vinv: $Vinv")

        # test stopping criterion
        if convex_sr
            Z, _, (star, ξ) = glrt(pep, P, hμ)
        else
            Z, (_, _), (star, ξ) = glrt(pep, N, hμ)
        end
        #println("star : $star; Z : $Z; beta : $(βs[1](t))")

        while Z > βs[1](t)
        #println("Z big")
            popfirst!(βs)
            push!(R, (star, copy(N), copy(P), CPUtime_us() - baseline))
            if isempty(βs)
                return R
            end
        end

        # invoke sampling rule
        i, k = nextsample(state, pep, star, ξ, N, P, S, Vinv)
        play!(i, k, rng, pep, µ, S, N, P, Vinv)
        convex_sr ? P[i, k] += 1 : nothing
        t += 1
    end
end

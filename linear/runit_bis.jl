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
    convex_sr = typeof(sr) == ConvexGame || typeof(sr) == ConvexGameL  # test if P is needed.
    gap_sr = typeof(sr) == LinGapE
    sfwt_sr = typeof(sr) == SLT3C
    sfwl_sr = typeof(sr) == SLGapE
    xya_sr = typeof(sr) == XYAdaptive

    βs = collect(βs) # mutable copy

    rng = MersenneTwister(seed)

    K = narms(pep, µs)
    nb_I = nanswers(pep, µs)
    dim = length(μs)
    convex_sr ? P = ones(Int64, (nb_I, K)) : P = ones(Int64, (1, 1))  # counts detailed by answer
    N = zeros(Int64, K)              # counts
    S = zeros(dim)                     # sum of samples
    Vinv = Matrix{Float64}(I, dim, dim)  # inverse of the design matrix

    sfwl_sr || sfwt_sr ? V = Matrix{Float64}(I, dim, dim) : nothing # counts matrix
    # sfwt_sr ? V = Matrix{Float64}(I, dim, dim) : nothing
    # sfwt_sr ? C = ones(K-1) : nothing
    xya_sr ? ρ = 1 : nothing
    xya_sr ? ρ_old = 1 : nothing
    xya_sr ? Xactive = copy(pep.arms) : nothing
    xya_sr ? α = 0.1 : nothing

    baseline = CPUtime_us()

    # pull each arm once. TODO: pull instead d arms spanning the space.
    for k = 1:K
        play!(1, k, rng, pep, µs, S, N, P, Vinv)
    end

    state = start(sr, N, P)

    t_old = sum(N)

    R = Tuple{Int64,Array{Int64,1},Array{Int64,2},UInt64}[] # collect return values

    while true
        t = sum(N)

        hµ = Vinv * S  # emp. estimates
        #println("hµ : $(round.(hµ, digits=3)); Vinv: $Vinv")

        # test stopping criterion
        if gap_sr
            _, (_, _), (star, ξ) = glrt(pep, N, hμ)

            # invoke sampling rule
            i, k, ucb = nextsample(state, pep, star, ξ, N, P, S, Vinv, βs[1](t))

            while ucb <= 0
            #println("B $B")
                popfirst!(βs)
                push!(R, (star, copy(N), copy(P), CPUtime_us() - baseline))
                if isempty(βs)
                    return R
                end
            end
        elseif  sfwl_sr
            _, (_, _), (star, ξ) = glrt(pep, N, hμ)

            # invoke sampling rule
            i, k, bnext, ucb = nextsample(state, pep, star, ξ, N, P, S, Vinv, V, βs[1](t))

            while ucb <= 0
            #println("B $B")
                popfirst!(βs)
                push!(R, (star, copy(N), copy(P), CPUtime_us() - baseline))
                if isempty(βs)
                    return R
                end
            end
        elseif sfwt_sr
            Z, (_, _), (star, ξ) = glrt(pep, N, hμ)

            # invoke sampling rule
            i, k, bnext = nextsample(state, pep, star, ξ, N, P, S, Vinv, V)

            while Z > βs[1](t)
            #println("Z big")
                popfirst!(βs)
                push!(R, (star, copy(N), copy(P), CPUtime_us() - baseline))
                if isempty(βs)
                    return R
                end
            end
        elseif xya_sr
            #_, (_, _), (star, ξ) = glrt(pep, N, hμ)

            # invoke sampling rule
            i, k, Xactive, ρ, ρ_old, t_old = nextsample(
                state,
                pep,
                N,
                P,
                S,
                Vinv,
                Xactive,
                α,
                ρ,
                ρ_old,
                t_old,
                βs[1](t),
            )

            while length(Xactive) <= 1
            #println("B $B")
                popfirst!(βs)
                push!(R, (i, copy(N), copy(P), CPUtime_us() - baseline))
                if isempty(βs)
                    return R
                end
            end
        else
            Z, (_, _), (star, ξ) = glrt(pep, N, hμ)

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
        end
        #println("star : $star; Z : $Z; beta : $(βs[1](t))")

        # play the choosen arm
        play!(i, k, rng, pep, µ, S, N, P, Vinv)
        sfwl_sr || sfwt_sr ? V .= V .+ bnext*transpose(bnext) : nothing
        # sfwt_sr ? V .= V .+ bnext*transpose(bnext) : nothing
        # sfwt_sr ? C[idb] += 1 : nothing
        convex_sr ? P[i, k] += 1 : nothing
        t += 1
    end
end

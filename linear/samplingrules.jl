# Sampling rules.
# We organise them in two levels
# - sampling rule; a factory for sampling rule states
# - sampling rule state; keeps track of i.e. tracking information etc.

include("../regret.jl");
include("../tracking.jl");
include("optimal_design.jl");


"""
Uniform sampling
"""

struct RoundRobin end

long(sr::RoundRobin) = "Uniform";
abbrev(sr::RoundRobin) = "RR";

function start(sr::RoundRobin, N, P)
    return sr
end

function nextsample(sr::RoundRobin, pep, star, ξ, N, P, S, Vinv)
    return (1 + (sum(N) % length(N))) * ones(Int64, 2)
end


"""
Tracking fixed weights
"""

struct FixedWeights # used as factory and state
    w
    function FixedWeights(w)
        @assert all(w .≥ 0) && sum(w) ≈ 1 "$w not in simplex"
        new(w)
    end
end

long(sr::FixedWeights) = "Fixed Weights";
abbrev(sr::FixedWeights) = "fix";

function start(sr::FixedWeights, N, P)
    return sr
end

function nextsample(sr::FixedWeights, pep, star, ξ, N, P, S, Vinv)
    argmin(N .- sum(N) .* sr.w) * ones(Int64, 2)
end


"""
Convexified game
"""

struct ConvexGame
    TrackingRule
end

long(sr::ConvexGame) = "ConvexGame " * abbrev(sr.TrackingRule);
abbrev(sr::ConvexGame) = "CG-" * abbrev(sr.TrackingRule);

struct ConvexGameState
    h  # one online learner
    t  # tracking rule
    ConvexGameState(TrackingRule, P) = new(AdaHedge(length(P)), TrackingRule(vec(P)))
end

function start(sr::ConvexGame, N, P)
    ConvexGameState(sr.TrackingRule, P)
end

# optimistic gradients
function optimistic_gradient(pep, hμ, t, P::Matrix, λs, Vinv)
    nb_I = nanswers(pep, hµ)
    K = narms(pep, hµ)
    grads = zeros(size(P))
    for k = 1:K
        arm = pep.arms[k]
        for i = 1:nb_I
            ref_value = (hµ .- λs[i, :])'arm
            confidence_width = log(t)
            deviation = sqrt(2 * confidence_width * ((arm') * Vinv * arm))
            ref_value > 0 ? grads[i, k] = 0.5 * (ref_value + deviation)^2 :
            grads[i, k] = 0.5 * (ref_value - deviation)^2
            grads[i, k] = min(grads[i, k], confidence_width)
        end
    end
    return grads
end

function nextsample(sr::ConvexGameState, pep, star, ξ, N, P, S, Vinv)
    nb_I = size(P)[1]
    K = size(P)[2]

    t = sum(N)

    hμ = Vinv * S # emp. estimates
    #println("hµ $hµ ; Vinv $Vinv")

    # query the learner
    vec_W = act(sr.h)
    W = permutedims(reshape(vec_W, (K, nb_I)), (2, 1))  # W is the I*K matrix of answers and pulls.

    # best response λ-player
    #println("Best response computation")
    #println("W $W")
    _, λs, (_, ξs) = glrt(pep, W, hμ)

    ∇ = vec(transpose(optimistic_gradient(pep, hμ, t, P, λs, Vinv)))
    #println("∇ $∇")

    incur!(sr.h, -∇)

    # tracking
    #println("Tracking")
    #println("P $P")
    #println("sumW $(sr.t.sumw .+ vec_W)")
    big_index = track(sr.t, vec(transpose(P)), vec_W)
    #println("track: $big_index")
    i = div(big_index - 1, K) + 1
    k = ((big_index - 1) % K) + 1
    #println("i $i k $k")
    return i, k
end


"""
Convexified game (be the leader)
"""

struct ConvexGameL
    TrackingRule
end

long(sr::ConvexGameL) = "ConvexGameL " * abbrev(sr.TrackingRule);
abbrev(sr::ConvexGameL) = "CGL-" * abbrev(sr.TrackingRule);

struct ConvexGameLState
    h  # one online learner
    t  # tracking rule
    ConvexGameLState(TrackingRule, P) = new(AdaHedge(length(P)), TrackingRule(vec(P)))
end

function start(sr::ConvexGameL, N, P)
    ConvexGameLState(sr.TrackingRule, P)
end

# optimistic gradients
function optimistic_gradient(pep, hμ, t, P::Matrix, λs, Vinv)
    nb_I = nanswers(pep, hµ)
    K = narms(pep, hµ)
    grads = zeros(size(P))
    for k = 1:K
        arm = pep.arms[k]
        for i = 1:nb_I
            ref_value = (hµ .- λs[i, :])'arm
            confidence_width = log(t)
            deviation = sqrt(2 * confidence_width * ((arm') * Vinv * arm))
            ref_value > 0 ? grads[i, k] = 0.5 * (ref_value + deviation)^2 :
            grads[i, k] = 0.5 * (ref_value - deviation)^2
            grads[i, k] = min(grads[i, k], confidence_width)
        end
    end
    return grads
end

function nextsample(sr::ConvexGameLState, pep, star, ξ, N, P, S, Vinv)
    nb_I = size(P)[1]
    K = size(P)[2]

    t = sum(N)

    hμ = Vinv * S # emp. estimates
    #println("hµ $hµ ; Vinv $Vinv")

    # query the learner
    vec_W = act(sr.h)
    W = permutedims(reshape(vec_W, (K, nb_I)), (2, 1))  # W is the I*K matrix of answers and pulls.
    # W = permutedims(reshape(sr.t.sumw .+ vec_W, (K, nb_I)), (2, 1))
    W .+= P

    # best response λ-player
    #println("Best response computation")
    #println("W $W")
    _, λs, (_, ξs) = glrt(pep, W, hμ)

    ∇ = vec(transpose(optimistic_gradient(pep, hμ, t, P, λs, Vinv)))
    #println("∇ $∇")

    incur!(sr.h, -∇)

    # tracking
    #println("Tracking")
    #println("P $P")
    #println("sumW $(sr.t.sumw .+ vec_W)")
    big_index = track(sr.t, vec(transpose(P)), vec_W)
    #println("track: $big_index")
    i = div(big_index - 1, K) + 1
    k = ((big_index - 1) % K) + 1
    #println("i $i k $k")
    return i, k
end


"""
k-Learner, not convexified
"""

struct LearnerK
    TrackingRule
end

long(sr::LearnerK) = "LearnerK " * abbrev(sr.TrackingRule);
abbrev(sr::LearnerK) = "Lk-" * abbrev(sr.TrackingRule);

struct LearnerKState
    hs  # I online learners
    t  # tracking rule
    LearnerKState(TrackingRule, N) = new(
        Dict{Int64,AdaHedge}(),  # We could allocate one AdaHedge for each answer, but for some problems there are 2^d answers.
        TrackingRule(vec(N)),
    )
end

function start(sr::LearnerK, N, P)
    LearnerKState(sr.TrackingRule, N)
end

# optimistic gradients
function optimistic_gradient(pep, hμ, t, N::Vector, λ, Vinv)
    K = length(pep.arms)
    grads = zeros(length(N))
    for k = 1:K
        arm = pep.arms[k]
        ref_value = (hµ .- λ)'arm
        confidence_width = log(t)
        deviation = sqrt(2 * confidence_width * ((arm') * Vinv * arm))
        ref_value > 0 ? grads[k] = 0.5 * (ref_value + deviation)^2 :
        grads[k] = 0.5 * (ref_value - deviation)^2
        grads[k] = min(grads[k], confidence_width)
    end
    return grads
end

function nextsample(sr::LearnerKState, pep, star, ξ, N, P, S, Vinv)
    t = sum(N)

    hμ = Vinv * S # emp. estimates
    #println("hµ $hµ ; Vinv $Vinv")

    nb_I = nanswers(pep, hµ)
    K = length(pep.arms)

    star = istar(pep, hµ)
    if !haskey(sr.hs, star)  # if we never saw that star, initialize an AdaHedge learner
        sr.hs[star] = AdaHedge(length(N))
    end

    # query the learner
    w = act(sr.hs[star])

    # best response λ-player
    #println("Best response computation")
    #println("w $w")
    _, (_, λ), (_, ξs) = glrt(pep, w, hμ)

    ∇ = optimistic_gradient(pep, hμ, t, N, λ, Vinv)
    #println("∇ $∇")

    incur!(sr.hs[star], -∇)

    # tracking
    k = track(sr.t, vec(N), w)
    #println("k $k")
    return star, k
end


"""
Track-and-Stop (Garivier and Kaufmann, 2016)
"""

struct TnS
    TrackingRule
end

long(sr::TnS) = "Track-and-Stop " * abbrev(sr.TrackingRule);
abbrev(sr::TnS) = "TnS-" * abbrev(sr.TrackingRule);

struct TnSState
    t  # tracking rule
    TnSState(TrackingRule, N) = new(ForcedExploration(TrackingRule(N)),)
end

function start(sr::TnS, N, P)
    TnSState(sr.TrackingRule, N)
end

function nextsample(sr::TnSState, pep, star, ξ, N, P, S, Vinv)
    t = sum(N)

    hμ = Vinv * S # emp. estimates
    #println("hµ $hµ ; Vinv $Vinv")

    nb_I = nanswers(pep, hµ)
    K = length(pep.arms)

    star = istar(pep, hµ)
    xstar = pep.arms[star]
    #@show xstar

    # compute the plug-in estimate
    items = pep.arms
    measures = build_T(items, xstar, hμ)
    w, _, _ = optimal_design_fw(items, measures, hμ)

    # tracking
    k = track(sr.t, vec(N), w)
    #println("k $k")
    return star, k
end


"""
k-Learner from DKM2019
"""

struct LearnerK_DKM
    TrackingRule
end

long(sr::LearnerK_DKM) = "LearnerK DKM " * abbrev(sr.TrackingRule);
abbrev(sr::LearnerK_DKM) = "LkDKM-" * abbrev(sr.TrackingRule);

struct LearnerK_DKMState
    hs  # I online learners
    t  # tracking rule
    LearnerK_DKMState(TrackingRule, N) = new(
        Dict{Int64,AdaHedge}(),  # We could allocate one AdaHedge for each answer, but for some problems there are 2^d answers.
        TrackingRule(vec(N)),
    )
end

function start(sr::LearnerK_DKM, N, P)
    LearnerK_DKMState(sr.TrackingRule, N)
end

# optimistic gradients
function optimistic_gradient_DKM(pep, hμ, t, N::Vector, λ)
    K = length(pep.arms)
    grads = zeros(length(N))
    for k = 1:K
        arm = pep.arms[k]
        ref_value = (hµ .- λ)'arm
        confidence_width = log(t)
        deviation = sqrt(2 * confidence_width / N[k])
        ref_value > 0 ? grads[k] = 0.5 * (ref_value + deviation)^2 :
        grads[k] = 0.5 * (ref_value - deviation)^2
        grads[k] = min(grads[k], confidence_width)
    end
    return grads
end

function nextsample(sr::LearnerK_DKMState, pep, star, ξ, N, P, S, Vinv)
    t = sum(N)

    hμ = Vinv * S # emp. estimates

    nb_I = nanswers(pep, hµ)
    K = length(pep.arms)

    star = istar(pep, hµ)
    if !haskey(sr.hs, star)  # if we never saw that star, initialize an AdaHedge learner
        sr.hs[star] = AdaHedge(length(N))
    end

    # query the learner
    w = act(sr.hs[star])

    # best response λ-player
    #println("Best response computation")
    #println("w $w")
    _, (_, λ), (_, ξs) = glrt(pep, w, hμ)

    ∇ = optimistic_gradient_DKM(pep, hμ, t, N, λ)
    #println("∇ $∇")

    incur!(sr.hs[star], -∇)

    # tracking
    k = track(sr.t, vec(N), w)
    #println("k $k")
    return star, k
end


"""
L-T3C
"""

struct LT3C
    ExplorationType
end

long(sr::LT3C) = "L-T3C-" * sr.ExplorationType;
abbrev(sr::LT3C) = "T-" * sr.ExplorationType;

function start(sr::LT3C, N, P)
    return sr
end

function compute_transportation(arm1, arm2, μ, cov)
    norm = transpose(arm1 - arm2) * cov * (arm1 - arm2)
    if (arm2'μ - arm1'μ < 0)
        cost = (arm1'μ - arm2'μ)^2 / (2 * norm)
    else
        cost = 0
    end
    return cost
end

function compute_confidence(arm1, arm2, cov)
    confidence = sqrt(transpose(arm1 - arm2) * cov * (arm1 - arm2))
    return confidence
end

function nextsample(sr::LT3C, pep, star, ξ, N, P, S, Vinv)
    hμ = Vinv * S # emp. estimates
    #println("hµ $hµ ; Vinv $Vinv")

    nb_I = nanswers(pep, hµ)
    K = length(pep.arms)
    dim = length(pep.arms[1])

    star = istar(pep, hµ)

    ts = zeros(K)
    for a = 1:K
        z = rand(MvNormal(dim, 1))
        θ = Vinv^0.5 * z + hμ
        ts[a] = sum(θ .* pep.arms[a])
    end
    best = argmax(ts)

    scores = zeros(K)
    for i = 1:K
        if i != best
            scores[i] = compute_transportation(pep.arms[best], pep.arms[i], hμ, Vinv)
        else
            scores[i] = Inf
        end
    end
    challenger = randmin(scores)

    if sr.ExplorationType == "G"
        k = randmin([compute_confidence(
            pep.arms[best],
            pep.arms[challenger],
            sherman_morrison(Vinv, pep.arms[i]),
        ) for i = 1:K])
    elseif sr.ExplorationType == "N"
        if (rand() > 0.5)
            k = best
        else
            k = challenger
        end
    end

    return star, k
end


"""
Saddle-point Frank-Wolfe L-T3C
"""

struct SLT3C end

long(sr::SLT3C) = "SL-T3C";
abbrev(sr::SLT3C) = "SFW-T";

function start(sr::SLT3C, N, P)
    return sr
end

function nextsample(sr::SLT3C, pep, star, ξ, N, P, S, Vinv, V)
    hμ = Vinv * S # emp. estimates
    #println("hµ $hµ ; Vinv $Vinv")

    nb_I = nanswers(pep, hµ)
    K = length(pep.arms)
    dim = length(pep.arms[1])

    star = istar(pep, hµ)

    ts = zeros(K)
    for a = 1:K
        z = rand(MvNormal(dim, 1))
        θ = Vinv^0.5 * z + hμ
        ts[a] = sum(θ .* pep.arms[a])
    end
    best = argmax(ts)
    Y = build_T(pep.arms, pep.arms[best], hμ)

    ida = argmax([2 * transpose(pep.arms[i]) * Vinv * V * Vinv * pep.arms[i] for i = 1:K])
    idb = argmax([transpose(b) * Vinv * b for b in Y])
    bnext = Y[idb]

    # L = length(Y)
    # ida = argmin([sum([C[i] * (-2) * (transpose(pep.arms[j]) * Vinv * Y[i]) .^ 2 for i = 1:L]) for j = 1:K])
    # idb = argmax([transpose(Y[i]) * Vinv * Y[i] for i = 1:L])

    return star, ida, bnext
end


"""
Saddle-point Frank-Wolfe LinGapE
"""

struct SLGapE end

long(sr::SLGapE) = "SLGapE";
abbrev(sr::SLGapE) = "SFW-L";

function start(sr::SLGapE, N, P)
    return sr
end

function gap(arm1, arm2, μ)
    (arm1 - arm2)'μ
end

function confidence(arm1, arm2, Vinv)
    sqrt(transpose(arm1 - arm2) * Vinv * (arm1 - arm2))
end

function nextsample(sr::SLGapE, pep, star, ξ, N, P, S, Vinv, V, β)
    hμ = Vinv * S # emp. estimates
    #println("hµ $hµ ; Vinv $Vinv")

    nb_I = nanswers(pep, hµ)
    K = length(pep.arms)
    dim = length(pep.arms[1])

    star = istar(pep, hµ)
    xstar = pep.arms[star]

    # Y = build_T(pep.arms, xstar, hμ)

    c_t = sqrt(2 * β)
    ucb, ambiguous = findmax([gap(pep.arms[i], xstar, hμ) +
                              confidence(pep.arms[i], xstar, Vinv) * c_t for i = 1:K])

    ida = argmax([2 * transpose(pep.arms[i]) * Vinv * V * Vinv * pep.arms[i] for i = 1:K])
    bnext = sqrt(2) * (xstar - pep.arms[ambiguous]) / (xstar - pep.arms[ambiguous])'hμ

    return star, ida, bnext, ucb
end


"""
LinGapE (Xu et al. 2018)
"""

struct LinGapE end

long(sr::LinGapE) = "LinGapE";
abbrev(sr::LinGapE) = "LG";

function start(sr::LinGapE, N, P)
    return sr
end

function gap(arm1, arm2, μ)
    (arm1 - arm2)'μ
end

function confidence(arm1, arm2, Vinv)
    sqrt(transpose(arm1 - arm2) * Vinv * (arm1 - arm2))
end

function nextsample(sr::LinGapE, pep, star, ξ, N, P, S, Vinv, β)
    hμ = Vinv * S # emp. estimates
    #println("hµ $hµ ; Vinv $Vinv")

    nb_I = nanswers(pep, hµ)
    K = length(pep.arms)

    star = istar(pep, hµ)

    c_t = sqrt(2 * β)
    ucb, ambiguous = findmax([gap(pep.arms[i], pep.arms[star], hμ) +
                              confidence(pep.arms[i], pep.arms[star], Vinv) * c_t for i = 1:K])

    k = argmin([confidence(
        pep.arms[star],
        pep.arms[ambiguous],
        sherman_morrison(Vinv, pep.arms[i]),
    ) for i = 1:K])

    return star, k, ucb
end


"""
XY-Allocation (Soare et al. 2014)
"""

struct Static
    DesignType
end

long(sr::Static) = sr.DesignType * "-Allocation"
abbrev(sr::Static) = sr.DesignType * "S"

function start(sr::Static, N, P)
    return sr
end

function nextsample(sr::Static, pep, star, ξ, N, P, S, Vinv)
    hμ = Vinv * S # emp. estimates
    #println("hµ $hµ ; Vinv $Vinv")

    nb_I = nanswers(pep, hµ)
    K = length(pep.arms)

    star = istar(pep, hµ)

    if sr.DesignType == "G"
        k = randmin([maximum([transpose(pep.arms[i]) * sherman_morrison(Vinv, pep.arms[j]) *
                              pep.arms[i] for i = 1:K]) for j = 1:K])
    elseif sr.DesignType == "XY"
        gaps = build_gaps(pep.arms)
        k = randmin([maximum([transpose(gaps[i]) * sherman_morrison(Vinv, pep.arms[j]) *
                              gaps[i] for i = 1:K]) for j = 1:K])
    end

    return star, k
end

struct XYAdaptive end

long(sr::XYAdaptive) = "XY-Adaptive";
abbrev(sr::XYAdaptive) = "XYA";

function start(sr::XYAdaptive, N, P)
    return sr
end

function drop_arms(Xactive, Vinv, μ, β)
    X = copy(Xactive)
    K = length(Xactive)
    for i = 1:K
        arm = X[i]
        for j = 1:K
            if j == i
                continue
            end
            arm_prime = X[j]
            y = arm_prime - arm
            # if y'μ < 0
            #    continue
            if (y' * Vinv * y * 2 * β)^0.5 <= y'μ
                filter!(x -> x ≠ arm, Xactive)
                break
            end
        end
    end
    return Xactive
end

function nextsample(sr::XYAdaptive, pep, N, P, S, Vinv, Xactive, α, ρ, ρ_old, t_old, β)
    t = sum(N)
    #@show t

    hμ = Vinv * S # emp. estimates
    #println("hµ $hµ ; Vinv $Vinv")
    #@show hμ, t

    nb_I = nanswers(pep, hµ)
    K = length(Xactive)

    star = istar(pep, hµ)

    Y = build_gaps(Xactive)
    nb_gaps = length(Y)
    #@show nb_gaps, t
    k = randmin([maximum([transpose(Y[i]) * sherman_morrison(Vinv, pep.arms[j]) * Y[i] for i = 1:nb_gaps]) for j = 1:nb_I])
    #@show k, t

    if ρ / t < α * ρ_old / t_old
        t_old = t
        ρ_old = ρ
        Xcopy = copy(pep.arms)
        Xactive = drop_arms(Xcopy, Vinv, hμ, β)
    end

    ρ = maximum([transpose(Y[i]) * sherman_morrison(Vinv, pep.arms[k]) * Y[i] for i = 1:nb_gaps])
    #@show ρ, t

    return star, k, Xactive, ρ, ρ_old, t_old
end


"""
RAGE (Fiez et al. 2019)
"""

struct RAGE end

long(sr::RAGE) = "RAGE";
abbrev(sr::RAGE) = "RA";

function start(sr::RAGE, N, P)
    return sr
end

function drop_arms(Xactive, Vinv, μ, β)
    X = copy(Xactive)
    K = length(Xactive)
    for i = 1:K
        arm = X[i]
        for j = 1:K
            if j == i
                continue
            end
            arm_prime = X[j]
            y = arm_prime - arm
            if y'μ < 0
                continue
            elseif y' * Vinv * y * 2 * β <= (y'μ)^2
                filter!(x -> x ≠ arm, Xactive)
                break
            end
        end
    end
    return Xactive
end

function nextsample(sr::RAGE, pep, N, P, S, Vinv)
    t = sum(N)
    #@show t

    hμ = Vinv * S # emp. estimates
    #println("hµ $hµ ; Vinv $Vinv")
    #@show hμ, t

    nb_I = nanswers(pep, hµ)
    K = length(Xactive)

    star = istar(pep, hµ)

    Y = build_gaps(Xactive)
    nb_gaps = length(Y)
    #@show nb_gaps, t
    k = randmin([maximum([transpose(Y[i]) * sherman_morrison(Vinv, pep.arms[j]) * Y[i] for i = 1:nb_gaps]) for j = 1:nb_I])
    #@show k, t

    if ρ / t < α * ρ_old / t_old
        t_old = t
        ρ_old = ρ
        Xcopy = copy(pep.arms)
        Xactive = drop_arms(Xcopy, Vinv, hμ, β)
    end

    ρ = maximum([transpose(Y[i]) * sherman_morrison(Vinv, pep.arms[k]) * Y[i] for i = 1:nb_gaps])
    #@show ρ, t

    return star, k, Xactive, ρ, ρ_old, t_old
end

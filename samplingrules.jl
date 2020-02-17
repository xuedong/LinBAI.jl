# libpurex:
# sampling rules

# We organise them in two levels
# - sampling rule; a factory for sampling rule states
# - sampling rule state; keeps track of i.e. tracking information etc.


include("regret.jl");



# Tracking helper types

struct CTracking
    sumw;
    CTracking(N) = new(
        Float64.(N)
    ); # makes a copy of the starting situation
end

abbrev(_::Type{CTracking}) = "C";

# add a weight vector and track it
function track(t::CTracking, N, w)
    @assert all(N .≤ t.sumw.+1) "N $N  sumw $(t.sumw)";
    @assert sum(N) ≈ sum(t.sumw);

    t.sumw .+= w;
    argmin(N ./ t.sumw);
end


struct DTracking
    DTracking(N) = new();
end

abbrev(_::Type{DTracking}) = "D";

function track(t::DTracking, N, w)
    argmin(N .- sum(N).*w);
end


# Wrapper to add forced exploration to a tracking rule
struct ForcedExploration
    t;
end

function track(fe::ForcedExploration, N, w)
    t = sum(N);
    K = length(N);
    undersampled = N .≤ sqrt(t) .- K/2;
    if any(undersampled)
        track(fe.t, N, undersampled/sum(undersampled));
    else
        track(fe.t, N, w);
    end
end



# optimistic gradients
function optimistic_gradient(pep, hμ, t, N, λs)
    # TODO: there may need to be log(log(t)) terms here
    # TODO: is it correct to put uppers and lowers also through ξs? (consensus: hell NO)

    [let dist = getexpfam(pep, k),
     ↑ = dup(dist, hμ[k], log(t)/N[k]),
     ↓ = ddn(dist, hμ[k], log(t)/N[k])
     max(d(dist, ↑, λs[k]), d(dist, ↓, λs[k]), log(t)/N[k])
     end
     for k in eachindex(hμ)];
end






struct RoundRobin # used as factory and state
end

long(sr::RoundRobin) = "Uniform";
abbrev(sr::RoundRobin) = "RR";

function start(sr::RoundRobin, N)
    return sr;
end

function nextsample(sr::RoundRobin, pep, istar, ξ, N, S)
    return 1+(sum(N) % length(N));
end




struct FixedWeights # used as factory and state
    w;
    function FixedWeights(w)
        @assert all(w .≥ 0) && sum(w) ≈ 1 "$w not in simplex";
        new(w)
    end
end

long(sr::FixedWeights) = "Oracle Weigths";
abbrev(sr::FixedWeights) = "opt";

function start(sr::FixedWeights, N)
    return sr;
end

function nextsample(sr::FixedWeights, pep, istar, ξ, N, S)
    argmin(N .- sum(N).*sr.w);
end




struct FictitiousPlay
end

long(sr::FictitiousPlay) = "Fictitious Play";
abbrev(sr::FictitiousPlay) = "FP";

struct FictitiousPlayState
    sum∇;
    FictitiousPlayState(K) = new(zeros(K));
end


function start(sr::FictitiousPlay, N)
    FictitiousPlayState(length(N));
end

function nextsample(sr::FictitiousPlayState, pep, istar, ξ, N, S)
    K = length(N);
    t = sum(N);
    hμ = S./N; # emp. estimates

    # best response λ-player to all past
    _, (_, λs), (_, ξs) = glrt(pep, N, hμ);

    #println("N =$N");
    #println("hμ=$hμ");
    #println("λs=$λs");
    #println("ξs=$ξs");

    ∇ = optimistic_gradient(pep, hμ, t, N, λs);
    sr.sum∇ .+= ∇;

    #println("$t, $N because $∇ summed to $(sr.sum∇)");

    argmax(sr.sum∇); # best response for k player to all past
end



struct TrackAndStop
    TrackingRule;
end

long(sr::TrackAndStop) = "Track-and-Stop " * abbrev(sr.TrackingRule);
abbrev(sr::TrackAndStop) = "T-" * abbrev(sr.TrackingRule);

struct TrackAndStopState
    t;
    TrackAndStopState(TrackingRule, N) = new(
        ForcedExploration(TrackingRule(N))
    );
end


function start(sr::TrackAndStop, N)
    TrackAndStopState(sr.TrackingRule, N);
end


function nextsample(sr::TrackAndStopState, pep, istar, ξ, N, S)
    K = length(N);
    t = sum(N);

    # oracle at ξ (the closest feasible bandit model)
    _, w = oracle(pep, ξ);

    # tracking
    return track(sr.t, N, w);
end









struct OptimisticTrackAndStop
    TrackingRule;
end

long(sr::OptimisticTrackAndStop) = "Optimistic TaS " * abbrev(sr.TrackingRule);
abbrev(sr::OptimisticTrackAndStop) = "O-" * abbrev(sr.TrackingRule);

struct OptimisticTrackAndStopState
    t;
    OptimisticTrackAndStopState(TrackingRule, N) = new(
        TrackingRule(N)
    );
end


function start(sr::OptimisticTrackAndStop, N)
    OptimisticTrackAndStopState(sr.TrackingRule, N);
end


function nextsample(sr::OptimisticTrackAndStopState, pep, istar, ξ, N, S)
    K = length(N);
    t = sum(N);

    # optimistic oracle at ξ (the closest feasible bandit model)
    # TODO: think about this ξ, should it be \hat μ? Or what?
    _, w = optimistic_oracle(pep, ξ, N);

    # tracking
    return track(sr.t, N, w);
end






struct Menard
    TrackingRule;
    scale;
end

long(sr::Menard) = "Menard " * abbrev(sr.TrackingRule);
abbrev(sr::Menard) = "M-" * abbrev(sr.TrackingRule);

function start(sr::Menard, N)
    MenardState(sr.TrackingRule, sr.scale, N);
end


struct MenardState
    h;
    t;
    MenardState(TrackingRule, scale, N) = new(
        FixedShare(length(N), S=scale),
        TrackingRule(N)
    );
end



function nextsample(sr::MenardState, pep, istar, ξ, N, S)
    K = length(N);
    t = sum(N);
    hμ = S./N; # emp. estimates

    # query the learner
    w = act(sr.h);

    # best response λ-player
    _, (_, λs), (_, _) = glrt(pep, w, hμ);

    # gradient
    ∇ = [d(getexpfam(pep, k), hμ[k], λs[k]) for k in eachindex(hμ)];

    # update learner
    incur!(sr.h, -∇);

    # tracking
    return track(sr.t, N, w);
end






struct UnaBomb
    TrackingRule;
end

long(sr::UnaBomb) = "UnaBomb " * abbrev(sr.TrackingRule);
abbrev(sr::UnaBomb) = "U-" * abbrev(sr.TrackingRule);

struct UnaBombState
    h; # one online learner in total
    t;
    UnaBombState(TrackingRule, N) = new(
        AdaHedge(length(N)),
        TrackingRule(N)
    );
end

function start(sr::UnaBomb, N)
    UnaBombState(sr.TrackingRule, N);
end


function nextsample(sr::UnaBombState, pep, istar, ξ, N, S)
    K = length(N);

    t = sum(N);

    hμ = S./N; # emp. estimates
    #println("hμ=$hμ");

    # query the learner
    w = act(sr.h);

    # best response λ-player
    _, (_, λs), (_, ξs) = glrt(pep, w, hμ);

    #println("λs=$λs");
    #println("ξs=$ξs");

    #println("↑s=$([dup(dist, hμ[k], log(t)/N[k]) for k in 1:K])");
    #println("↓s=$([ddn(dist, hμ[k], log(t)/N[k]) for k in 1:K])");

    # feed linear loss plus optimism back to learner
    # println("d=$([d(dist, hμ[k], λs[k]) for k in 1:K])");

    # TODO: there may need to be log(log(t)) terms here
    # TODO: is it correct to put uppers and lowers also through ξs?
    ∇ = optimistic_gradient(pep, hμ, t, N, λs);
    #println("∇=$∇");
    incur!(sr.h, -∇);

    # tracking
    return track(sr.t, N, w);
end




struct DaBomb
    TrackingRule;
    M;
end

long(sr::DaBomb) = "DaBomb " * abbrev(sr.TrackingRule);
abbrev(sr::DaBomb) = "D-" * abbrev(sr.TrackingRule);

struct DaBombState
    hs; # one online learner per answer
    t;
    DaBombState(TrackingRule, N, M) = new(
        map(x -> AdaHedge(length(N)), 1:M),
        TrackingRule(N)
    );
end

function start(sr::DaBomb, N)
    DaBombState(sr.TrackingRule, N, sr.M);
end


function nextsample(sr::DaBombState, pep, istar, ξ, N, S)
    K = length(N);

    t = sum(N);

    hμ = S./N; # emp. estimates
    #println("hμ=$hμ");

    # query the learner
    w = act(sr.hs[istar]);

    # best response λ-player
    _, (_, λs), (_, ξs) = glrt(pep, w, hμ);

    #println("λs=$λs");
    #println("ξs=$ξs");

    #println("↑s=$([dup(dist, hμ[k], log(t)/N[k]) for k in 1:K])");
    #println("↓s=$([ddn(dist, hμ[k], log(t)/N[k]) for k in 1:K])");

    # feed linear loss plus optimism back to learner
    # println("d=$([d(dist, hμ[k], λs[k]) for k in 1:K])");

    ∇ = optimistic_gradient(pep, hμ, t, N, λs)

    #println("∇=$∇");
    incur!(sr.hs[istar], -∇);

    # tracking
    return track(sr.t, N, w);
end

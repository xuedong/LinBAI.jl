# libpurex:
# oracle computations


# a Pure Exploration problem (pep) is parameterised by
# - a domain, \mathcal M, representing the prior knowledge about the structure
# - a query, as embodied by a correct-answer function istar
#
# To specify a PE problem here, we need to compute the following things:
# - nanswers: number of possible answers
# - istar: correct answer for feasible μ
# - glrt: value and best response (λ and ξ) to (N, ̂μ) or (w, μ)
# - oracle: characteristic time and oracle weights at μ


struct BestArm
    expfam; # common exponential family
end

nanswers(pep::BestArm, μ) = length(μ);
istar(pep::BestArm, μ) = argmax(μ);
getexpfam(pep::BestArm, k) = pep.expfam;

function glrt(pep::BestArm, w, μ)
    @assert length(size(μ)) == 1

    ⋆ = argmax(μ); # index of best arm among μ

    val, k, θ = minimum(
        begin
        # transport ⋆ and k to weighted midpoint
        θ = (w[⋆]*μ[⋆]+w[k]*μ[k])/(w[⋆]+w[k]);
        w[⋆]*d(pep.expfam, μ[⋆], θ) + w[k]*d(pep.expfam, μ[k], θ), k, θ
        end
        for k in eachindex(μ)
        if k != ⋆
    );

    λ = copy(μ);
    λ[⋆] = θ;
    λ[k] = θ;

    # note: ξs = μ here

    val, (k, λ), (⋆, μ);
end


# solve for x such that d(μ1, μx) + x*d(μi, μx) == v
# where μx = (μ1+x*μi)/(1+x)
function X(expfam, μ1, μi, v)
    kl1i = d(expfam, μ1, μi); # range of V(x) is [0, kl1i]
    @assert 0 ≤ v ≤ kl1i "0 ≤ $v ≤ $kl1i";
    α = binary_search(
        z -> let μz = (1-z)*μ1+z*μi
        (1-z)*d(expfam, μ1, μz) + z*d(expfam, μi, μz) - (1-z)*v
        end,
        0, 1);
    α/(1-α);
end


# oracle problem
function oracle(pep::BestArm, μs)
    μstar = maximum(μs);

    if all(μs .== μstar) # yes, this happens
        return Inf, ones(length(μs))/length(μs);
    end

    # determine upper range for subsequent binary search
    hi = minimum(
        d(pep.expfam, μstar, μ)
        for μ in μs
        if μ != μstar
    );

    val = binary_search(
        z -> sum(
            let x = X(pep.expfam, μstar, μ, z),
               μx = (μstar + x*μ)/(1+x);
            d(pep.expfam, μstar, μx) / d(pep.expfam, μ, μx)
            end
            for μ in μs
            if μ != μstar
            ) - 1,
        0, hi);

    ws = [(μ == μstar) ? 1. : X(pep.expfam, μstar, μ, val) for μ in μs];
    Σ = sum(ws);
    Σ/val, ws ./ Σ;
end


# oracle problem for best μ in confidence interval
function optimistic_oracle(pep::BestArm, hμ, N)

    t = sum(N);

    μdn = [ddn(pep.expfam, hμ[k], log(t)/N[k]) for k in eachindex(hμ)];
    μup = [dup(pep.expfam, hμ[k], log(t)/N[k]) for k in eachindex(hμ)];

    # try to make each arm the best-looking arm so far,
    # then move everybody else down as far as possible
    minimum(oracle(pep,
                   ((k == j) ? μup[k] : μdn[k]
                    for k in eachindex(hμ)))
            for j in eachindex(hμ)
            if μup[j] > maximum(μdn)
            );
end




struct LargestProfit
    expfam;
end

nanswers(pep::LargestProfit, μ) = length(μ)/2;

function istar(pep::LargestProfit, _μs)
    μs = reshape(_μs, 2, :);
    argmax(μs[1,:]-μs[2,:]); # current best
end
getexpfam(pep::LargestProfit, k) = pep.expfam;


# NOTE this does NOT work for divergences that are non-convex in the
# second argument. Moreover, the problem here seems fundamental.
function glrt(pep::LargestProfit, _ws, _μs)

    # convert to 2×K indexing
    ws = reshape(_ws, 2, :);
    μs = reshape(_μs, 2, :);

    K = size(ws, 2);

    ⋆ = argmax(μs[1,:]-μs[2,:]); # current best

    val, k, λ1k, λ2k, λ1s, λ2s = minimum(
        begin
        # arm k needs to be the best

        xmax = 1000; # voodoo
        x = binary_search(x ->
                          + invh(pep.expfam, μs[1,k], +x/ws[1,k])
                          - invh(pep.expfam, μs[2,k], -x/ws[2,k])
                          - invh(pep.expfam, μs[1,⋆], -x/ws[1,⋆])
                          + invh(pep.expfam, μs[2,⋆], +x/ws[2,⋆]), 0, xmax);

        λ1k = invh(pep.expfam, μs[1,k], +x/ws[1,k]);
        λ2k = invh(pep.expfam, μs[2,k], -x/ws[2,k]);
        λ1s = invh(pep.expfam, μs[1,⋆], -x/ws[1,⋆]);
        λ2s = invh(pep.expfam, μs[2,⋆], +x/ws[2,⋆]);

        ws[1,⋆]*d(pep.expfam, μs[1,⋆], λ1s) +
        ws[2,⋆]*d(pep.expfam, μs[2,⋆], λ2s) +
        ws[1,k]*d(pep.expfam, μs[1,k], λ1k) +
        ws[2,k]*d(pep.expfam, μs[2,k], λ2k), k, λ1k, λ2k, λ1s, λ2s
        end
        for k in 1:K
        if k != ⋆
    );
    _λs = copy(_μs);
    λs = reshape(_λs, 2, :);
    λs[1,⋆] = λ1s;
    λs[2,⋆] = λ2s;
    λs[1,k] = λ1k;
    λs[2,k] = λ2k;
    # note: ξs = μs here
    val, (k, _λs), (⋆, _μs);
end


function oracle(pep::LargestProfit, μs)
    @warn "oracle for LargestProfit not yet implemented";
    nothing, nothing;
end


struct MinimumThreshold
    expfam;
    γ;
end

nanswers(pep::MinimumThreshold, μ) = 2;
istar(pep::MinimumThreshold, μs) = 1+(minimum(μs) > pep.γ);
getexpfam(pep::MinimumThreshold, k) = pep.expfam;

# answers are encoded as 1 = "<", 2 = ">"
function glrt(pep::MinimumThreshold, ws, μs)
    if minimum(μs) < pep.γ
        #everybody moves up
        λs = max.(μs, pep.γ);
        val = sum(k -> ws[k]*d(pep.expfam, μs[k], λs[k]), eachindex(ws));
        val, (2, λs), (1, μs);
    else
        #someone moves down
        val, k = minimum(
            (ws[k]*d(pep.expfam, μs[k], pep.γ), k)
            for k in eachindex(ws));
        λs = copy(μs);
        λs[k] = pep.γ;
        val, (1, λs), (2, μs);
    end
end


function oracle(pep::MinimumThreshold, μs)
    μstar = minimum(μs);
    if μstar < pep.γ
        # everything on lowest arm
        ws = Float64.(μs .== μstar);
        Tstar = 1/d(pep.expfam, μstar, pep.γ);
    else
        ws = [1/d(pep.expfam, μ, pep.γ) for μ in μs];
        Tstar = sum(ws);
    end
    Tstar, ws ./ sum(ws);
end


# oracle problem for best μ in confidence interval
function optimistic_oracle(pep::MinimumThreshold, hμ, N)

    t = sum(N);

    μdn = (ddn(pep.expfam, hμ[k], log(t)/N[k]) for k in eachindex(hμ));
    μup = (dup(pep.expfam, hμ[k], log(t)/N[k]) for k in eachindex(hμ));

    if minimum(μup) < pep.γ
        # entire conf. interval below
        oracle(pep, μdn);
    elseif minimum(μdn) > pep.γ
        # entire conf. interval above
        oracle(pep, μup);
    else
        # both options in conf. interval. Take best
        min(oracle(pep, μup),
            oracle(pep, μdn));
    end
end


# Best arm with a simplex constraint on μ
struct BestArmSimplex
end




# all arms on same side of threshold
# interesting because ̂μ can be outside
struct SameSign
    expfam;
    γ; # threshold
end


nanswers(pep::SameSign, μ) = 2;
# answer 1 is everyone below,
# answer 2 is everyone above

function istar(pep::SameSign, μ)
    if all(μ .≤ pep.γ)
        1
    elseif all(μ .≥ pep.γ)
        2
    else
        @assert false "infeasible bandit model $μ";
    end
end
getexpfam(pep::SameSign, k) = pep.expfam;

function glrt(pep::SameSign, w, μ)
    μlo = min.(pep.γ, μ); # lower everyone
    μhi = max.(pep.γ, μ); # raise everyone
    dlo = sum(k -> w[k]*d(pep.expfam, μ[k], μlo[k]), eachindex(μ));
    dhi = sum(k -> w[k]*d(pep.expfam, μ[k], μhi[k]), eachindex(μ));

    if dlo < dhi
        dhi-dlo, (2, μhi), (1, μlo);
    else
        dlo-dhi, (1, μlo), (2, μhi);
    end
end

function oracle(pep::SameSign, μ)
    @assert all(μ .≤ pep.γ) || all(μ .≥ pep.γ) "infeasible bandit model $μ";

    v, i = findmax([d(pep.expfam, μ[k], pep.γ) for k in eachindex(μ)]);
    1/v, Float64.(eachindex(μ).==i)
end



# two arms with equal means but different exponential family (e.g. variance).
# question if they are above/below the threshold γ
# do algorithms pull the low variance one too often?
struct HeteroSkedastic
    expfams; # two(!) exponential families
    γ;
end


nanswers(pep::HeteroSkedastic, μ) = 2;
function istar(pep::HeteroSkedastic, μ)
    @assert length(μ) == 2;
    @assert abs(μ[1] - μ[2]) < 1e-4 "unequal $μ";
    1 + (μ[1] > pep.γ);
end

getexpfam(pep::HeteroSkedastic, k) = pep.expfams[k];


function glrt(pep::HeteroSkedastic, w, μ)

    # compute ξ
    xmax = 400; # voodoo
    if minimum(w) > 1e-5
        x = binary_search(x ->
                          + invh(pep.expfams[1], μ[1], +x/w[1])
                          - invh(pep.expfams[2], μ[2], -x/w[2]), -xmax, xmax);

        ξ = [invh(pep.expfams[1], μ[1], +x/w[1]),
             invh(pep.expfams[2], μ[2], -x/w[2])];
    else
        ξ = [w'μ, w'μ]; # two copies of the average
    end

    ⋆ = istar(pep, ξ);
    λ = [pep.γ, pep.γ];
    val = sum(k -> w[k]*(d(pep.expfams[k], μ[k], λ[k]) -
                         d(pep.expfams[k], μ[k], ξ[k])), 1:2);
    val, (3-⋆, λ), (⋆, ξ);
end

function oracle(pep::HeteroSkedastic, μ)
    v, i = findmax([d(pep.expfams[k], μ[k], pep.γ) for k in eachindex(μ)]);
    1/v, Float64.(eachindex(μ).==i)
end

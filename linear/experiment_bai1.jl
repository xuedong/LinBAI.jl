# Example from Soare et al.

using JLD2;
using Distributed;
using Printf;
using IterTools;
using Distributions

@everywhere include("runit_bis.jl");
@everywhere include("../thresholds.jl");
include("../experiment_helpers.jl");


# setup

dist = Gaussian();
dim = 2
μ = zeros(dim);
µ[1] = 1.0;
# μ = [.9, .7, .5];

arms = Vector{Float64}[]
for k = 1:dim
    v = zeros(dim)
    v[k] = 1.0
    push!(arms, v)
end
ω = pi / 6
v = zeros(dim);
v[1] = cos(ω);
v[2] = sin(ω);
push!(arms, v)

pep = LinearBestArm(dist, arms);

K = length(arms)
spanning_weights = ones(K)
spanning_weights[K] = 0.0
spanning_weights /= sum(spanning_weights)

# srs = [
#     ConvexGame(CTracking),
#     LearnerK(CTracking),
#     RoundRobin(),
#     FixedWeights(spanning_weights),
#     Static("G"),
#     Static("XY"),
#     XYAdaptive(),
#     LinGapE(),
# ]
srs = [SLGapE("TS"), LinGapE(), ConvexGame(CTracking)]


# δs = (0.1, 0.01, 0.0001);
δs = (0.00001,);
βs = GK16.(δs);

repeats = 100;
seed = 1234;


# compute

@time data = map(  # TODO: replace by pmap (it is easier to debug with map)
    ((sr, i),) -> runit(seed + i, sr, μ, pep, βs, δs),
    Iterators.product(srs, 1:repeats),
);

dump_stats(pep, μ, δs, βs, srs, data, repeats);


# save

@save isempty(ARGS) ? "experiment_bai1.dat" : ARGS[1] dist μ pep srs data δs βs repeats seed

# visualise by loading viz_bai1.jl

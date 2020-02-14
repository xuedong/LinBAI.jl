# Example from Tao et al.

using JLD2;
using Distributed;
using Printf;
using IterTools;
using Random;

@everywhere include("runit_bis.jl");
@everywhere include("../thresholds.jl");
include("../experiment_helpers.jl");


rng = MersenneTwister(1234)

# setup

dist = Gaussian();
dim = 12
num_arms = 20

# Box-Mueller
arms = Vector{Float64}[]
for k = 1:num_arms
    v = zeros(dim)
    for i = 1:dim
        v[i] = randn()
    end
    normalise = norm(v)
    v ./= norm(v)
    push!(arms, v)
end

μ = arms[1] + 0.01 * (arms[2] - arms[1]);

pep = LinearBestArm(dist, arms);

K = length(arms)
spanning_weights = ones(K)
spanning_weights[K] = 0.0
spanning_weights /= sum(spanning_weights)

srs = [
    ConvexGame(CTracking),
    LearnerK(CTracking),
    RoundRobin(),
    FixedWeights(spanning_weights),
    Static("G"),
    Static("XY"),
    XYAdaptive(),
    LinGapE(),
]
# srs = [LinGapE()]


# δs = (0.1, 0.01, 0.0001);
δs = (0.01,);
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

@save isempty(ARGS) ? "experiment_bai2.dat" : ARGS[1] dist μ pep srs data δs βs repeats seed

# visualise by loading viz_bai2.jl

# replicate the first experiment from Optimal Best Arm Identification
# with Fixed Confidence (Garivier and Kaufmann 2016).
#
# running on my laptop this takes
# 931.314152 seconds (9.09 M allocations: 1.541 GiB, 0.03% gc time)

using JLD2;
using Distributed;
using Printf;

@everywhere include("runit.jl");
@everywhere include("thresholds.jl");
include("experiment_helpers.jl");

# setup

dist = Bernoulli();
μ = [.5, .45, .43, .4];
pep = BestArm(dist);

srs = everybody(pep, μ);


δs = (0.5, 0.1);
βs = GK16.(δs); # Recommended in section 6 of paper

N = 3000;
seed = 1234;


# compute

@time data = pmap(
    ((sr,i),) -> runit(seed+i, sr, μ, pep, βs),
    Iterators.product(srs, 1:N)
);

dump_stats(pep, μ, δs, βs, srs, data);


# save

@save isempty(ARGS) ? "experiment_bai1.dat" : ARGS[1]  dist μ pep srs data δs βs N seed

# visualise by loading viz_bai1.jl

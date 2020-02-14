# Thresholding bandit with d arms: the basis vectors.

using JLD2;
using Distributed;
using Printf;

@everywhere include("runit.jl");
@everywhere include("../thresholds.jl");
include("../experiment_helpers.jl");


# setup

dist = Gaussian();
dim = 3
μ = zeros(dim); µ[1] = 1.;

arms = Vector{Float64}[]
for k in 1:dim
	v = zeros(dim); v[k] = 1.
	push!(arms, v)
end
#ω = pi/6
#v = zeros(dim); v[1] = cos(ω); v[2] = sin(ω)
#push!(arms, v)

τ = 0.5

pep = LinearThreshold(dist, arms, τ);

srs = [RoundRobin(), LearnerK(CTracking)]


δs = (0.1, 0.01, 0.0001);
βs = GK16.(δs);

repeats = 10;
seed = 1234;


# compute

@time data = map(  # TODO: replace by pmap (it is easier to debug with map)
    ((sr,i),) -> runit(seed+i, sr, μ, pep, βs),
    Iterators.product(srs, 1:repeats)
);

dump_stats(pep, μ, δs, βs, srs, data, repeats);


# save

@save isempty(ARGS) ? "experiment_threshold1.dat" : ARGS[1]  dist μ pep srs data δs βs repeats seed

# visualise by loading viz_bai1.jl

using JLD2;
using Printf;
using StatsPlots;

include("runit.jl"); # for types
include("../experiment_helpers.jl");
include("../thresholds.jl");

name = "experiment_threshold1";

@load "$name.dat" dist μ pep srs data δs βs repeats seed

dump_stats(pep, μ, δs, βs, srs, data, repeats);

for i in 1:length(δs)
    plot(boxes(pep, μ, δs[i], βs[i], srs, getindex.(data, i)));
    savefig("$(name)_$i.pdf");
end

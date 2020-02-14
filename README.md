To run an experiment, run julia in the "linear" subfolder and do
julia> include("experiment_XXXX.jl")

To generate plots once the experiment saved a experiment_XXX.dat file, run
julia> include("viz_XXXX.jl")
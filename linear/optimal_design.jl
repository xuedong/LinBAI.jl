using LinearAlgebra
using Printf
include("pep_linear.jl");
include("../expfam.jl");


function optimal_design_fw_aa(
    items,
    measures,
    max_iter = 50000,
    early_stopping = 1e-5,
    drop = false,
    fix_threshold = 1e-5,
)
    K = length(items[1, :])
    L = length(measures[:, 1])
    design = ones(length(items[:, 1]))
    design /= sum(design)
    Vinv = Matrix{Float64}(I, K, K)

    for count = 1:max_iter
        idx = randmin([maximum([transpose(measures[i, :]) *
                                sherman_morrison(Vinv, items[j, :]) * measures[i, :] for i = 1:L]) for j = 1:K])

        Vinv = sherman_morrison(Vinv, items[idx, :])

        γ = 1 / (count + 1)
        design_update = -γ * design
        design_update[idx] += γ

        relative = norm(design_update) / norm(design)

        design += design_update

        if relative < early_stopping
            println("early stopped")
            break
        end
    end

    if drop
        drop_total = sum(filter(x -> x < fix_threshold, design))
        if drop_total > 0
            design = replace(x -> x < fix_threshold ? 0 : x, design)
            design[argmax(design)] += drop_total
        end
    end

    return design
end


function optimal_design_fw_ab(
    items,
    measures,
    μ,
    max_iter = 10000,
    early_stopping = 1e-5,
    drop = false,
    fix_threshold = 1e-5,
)
    K = length(items[1, :])
    L = length(measures[:, 1])
    design = ones(length(items[:, 1]))
    design /= sum(design)
    measure_counts = ones(L)
    Vinv = Matrix{Float64}(I, K, K)

    for count = 1:max_iter
        ida = argmin([sum([measure_counts[i] * (-2) *
                           (transpose(items[j, :]) * Vinv * measures[i, :]) .^ 2 for i = 1:L]) for j = 1:K])
        # r = [measure_counts[i] * (-2) * transpose(items[1]) * Vinv * measures[i, :] for i = 1:L]
        # @show r
        idb = argmax([transpose(measures[i, :]) * Vinv * measures[i, :] for i = 1:L])
        # s = [transpose(measures[i, :]) * Vinv * measures[i, :] for i = 1:L]
        # @show r, idb, ida

        Vinv = sherman_morrison(Vinv, items[ida, :])
        measure_counts[idb] += 1

        γ = 1 / (count + 1)
        design_update = -γ * design
        design_update[ida] += γ

        relative = norm(design_update) / norm(design)

        design += design_update

        if relative < early_stopping
            println("early stopped")
            break
        end
    end

    if drop
        drop_total = sum(filter(x -> x < fix_threshold, design))
        if drop_total > 0
            design = replace(x -> x < fix_threshold ? 0 : x, design)
            design[argmax(design)] += drop_total
        end
    end

    # compute the complexity
    AB = maximum([transpose(measures[i, :]) * Vinv * measures[i, :] for i = 1:L])
    value = maximum([2 * transpose(items[1, :] - items[i, :]) * Vinv *
                     (items[1, :] - items[i, :]) / (items[1, :] - items[i, :])'μ for i = 2:K])

    return design, AB, value
end


function optimal_design_fw(
    items,
    measures,
    μ,
    max_iter = 10000,
    early_stopping = 1e-5,
    drop = false,
    fix_threshold = 1e-5,
)
    K = length(items[1])
    L = length(measures)
    design = ones(length(items))
    design /= sum(design)
    measure_counts = ones(L)
    Vinv = Matrix{Float64}(I, K, K)

    for count = 1:max_iter
        ida = argmin([sum([measure_counts[i] * (-2) *
                           (transpose(items[j]) * Vinv * measures[i]) .^ 2 for i = 1:L]) for j = 1:K])
        idb = argmax([transpose(measures[i]) * Vinv * measures[i] for i = 1:L])

        Vinv = sherman_morrison(Vinv, items[ida])
        measure_counts[idb] += 1

        γ = 1 / (count + 1)
        design_update = -γ * design
        design_update[ida] += γ

        relative = norm(design_update) / norm(design)

        design += design_update

        if relative < early_stopping
            println("early stopped")
            break
        end
    end

    if drop
        drop_total = sum(filter(x -> x < fix_threshold, design))
        if drop_total > 0
            design = replace(x -> x < fix_threshold ? 0 : x, design)
            design[argmax(design)] += drop_total
        end
    end

    # compute the complexity
    AB = maximum([transpose(measures[i]) * Vinv * measures[i] for i = 1:L])
    value = maximum([2 * transpose(items[1] - items[i]) * Vinv *
                     (items[1] - items[i]) / (items[1] - items[i])'μ for i = 2:K])

    return design, AB, value
end


function build_XY_ab(items)
    num_items = size(items)[1]
    num_features = size(items)[2]
    Y = zeros((num_items * num_items, num_features))

    for i = 1:num_items
        Y[(num_items*(i-1)+1):num_items*i, :] = items .- items[i, :]'
    end

    return Y
end


function build_XY(items)
    num_items = length(pep.arms)
    num_features = length(pep.arms[1])

    Y = Vector{Float64}[]
    for i = 1:num_items
        for j = 1:num_items
            v = items[j] - items[i]
            push!(Y, v)
        end
    end

    return Y
end


function build_T_ab(items, xstar, θ)
    num_items = size(items)[1]
    num_features = size(items)[2]
    Y = zeros((num_items - 1, num_features))

    for i = 1:(num_items-1)
        Y[i, :] = sqrt(2) * (xstar - items[i+1, :]) / (xstar - items[i+1, :])'θ
    end

    return Y
end


function build_T(items, xstar, θ)
    num_items = length(items)
    num_features = length(items[1])

    Y = Vector{Float64}[]
    for i = 1:(num_items-1)
        v = sqrt(2) * (xstar - items[i+1]) / (xstar - items[i+1])'θ
        push!(Y, v)
    end

    return Y
end


"""
Tests
"""
# dist = Gaussian();
# dim = 4
# μ = zeros(dim);
# µ[1] = 0.5; μ[2] = 0.45; μ[3] = 0.43; μ[4] = 0.4
#
# arms = Vector{Float64}[]
# for k = 1:dim
#     v = zeros(dim)
#     v[k] = 1.0
#     push!(arms, v)
# end
# # ω = pi / 6
# # v = zeros(dim);
# # v[1] = cos(ω);
# # v[2] = sin(ω);
# # push!(arms, v)
#
# pep = LinearBestArm(dist, arms);
# designAA, _, _ = optimal_design_fw(pep.arms, pep.arms, μ)
# designABdir, _, _ = optimal_design_fw(pep.arms, build_XY(pep.arms), μ)
# designABstar, _, _ = optimal_design_fw(pep.arms, build_T(pep.arms, pep.arms[1], μ), μ, 50000)
# @show designABstar

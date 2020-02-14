using LinearAlgebra
using Printf


function sherman_morrison(Vinv, u)
    Vinv_u = Vinv*u
    num = Vinv_u*transpose(Vinv_u)
    denum = 1 + transpose(u)*Vinv_u
    return Vinv .- num / denum
end


function optimal_design_fw(
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


function optimal_design_fw2(
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


function build_XY(items)
    num_items = size(items)[1]
    num_features = size(items)[2]
    Y = zeros((num_items * num_items, num_features))

    for i = 1:num_items
        Y[(num_items*(i-1)+1):num_items*i, :] = items .- items[i, :]'
    end

    return Y
end


function build_T(items, xstar, θ)
    num_items = size(items)[1]
    num_features = size(items)[2]
    Y = zeros((num_items - 1, num_features))

    for i = 1:(num_items-1)
        Y[i, :] = sqrt(2) * (xstar - items[i+1, :]) / (xstar - items[i+1, :])'θ
    end

    return Y
end


num_items = 3
num_features = 2

items = zeros(num_items, num_features)
for k = 1:num_features
    items[k, k] = 1.0
end
ω = pi / 4
items[num_items, 1] = cos(ω);
items[num_items, 2] = sin(ω);

measures1 = build_XY(items)
#@show measures1

#xstar = [1.0, 0.0, 0.0, 0.0]
#μ = [0.5, 0.45, 0.43, 0.4]
#xstar = [1.0, 0.0, 0.0, 0.0, 0.0]
#μ = [0.3, 0.21, 0.2, 0.19, 0.18]
xstar = [1.0, 0.0]
μ = [1.0, 0.0]

measures2 = build_T(items, xstar, μ)
#@show measures2

designAA, _, valueAA = optimal_design_fw2(items, items, μ)
designABdir, _, valueABdir = optimal_design_fw2(items, measures1, μ)
designABstar, _, valueABstar = optimal_design_fw2(items, measures2, μ, 50000)

@show designABstar

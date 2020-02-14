function sherman_morrison(Vinv, u, v)
    num = (Vinv*u)*transpose(transpose(Vinv)*v)
    denum = 1 + transpose(v)*Vinv*u
    return Vinv .- num / denum
end
function sherman_morrison(Vinv, u)
    Vinv_u = Vinv*u
    num = Vinv_u*transpose(Vinv_u)
    denum = 1 + transpose(u)*Vinv_u
    return Vinv .- num / denum
end

function build_gaps(arms)
    gaps = Vector{Float64}[]
    for pair in subsets(arms, 2)
        gap1 = pair[1] - pair[2]
        push!(gaps, gap1)
        gap2 = pair[2] - pair[1]
        push!(gaps, gap2)
        #@show pair
    end
    return gaps
end

struct LinearBestArm
    expfam; # common exponential family
    arms;  # array of size K of arms in R^d
end

nanswers(pep::LinearBestArm, μ) = length(pep.arms);
narms(pep::LinearBestArm, µ) = length(pep.arms);
istar(pep::LinearBestArm, μ) = argmax([arm'µ for arm in pep.arms]);
armstar(pep::LinearBestArm, μ) = pep.arms[argmax([arm'µ for arm in pep.arms])];
getexpfam(pep::LinearBestArm, k) = pep.expfam;

function alt_min(pep::LinearBestArm, w, µ, k)
    #TODO: this function assumes that expfam is Gaussian(1)
    #println("alt_min for k=$k, w=$w and µ=$µ")
    sum_w = sum(w)
    #if sum_w < 1e-10
    #    return 0, µ, k
    #end
    w = w/sum(w)  # avoid dividing by small quantities
    #println("w : $w")
    arm = pep.arms[k]
    arm_star = armstar(pep, µ)
    @assert arm != arm_star
    K = length(pep.arms)
    direction = arm .- arm_star
    #println("dir $direction")
    d = length(direction)
    # Construction of the matrix V = (sum_k w_k a_k a_k^T)^{-1}
    sum_arms_matrix = zeros(d,d)
    for j in 1:K
        sum_arms_matrix .+= w[j].* (pep.arms[j]*transpose(pep.arms[j]))
    end
    Vinv = inv(sum_arms_matrix)
    #println("Vinv $Vinv")
    # Closest point
    η = sum_w * (direction'µ) / ((direction')*Vinv*direction)
    #println("η $η; sum_w $sum_w")
    λ = µ .- η/sum_w * Vinv * direction
    # Divergence to that point
    val = .5 * sum_w * (direction'µ)^2 / ((direction')*Vinv*direction)
    #println("k $k; val $val; λ $λ")
    return val, λ, k
end

function alt_min(pep::LinearBestArm, w, µ)
    minimum(alt_min(pep, w, µ, i) for i in 1:nanswers(pep, µ) if i != istar(pep, µ))
end

function glrt(pep::LinearBestArm, w::Vector, μ)
    #println("glrt vector")
    @assert length(size(μ)) == 1
    star = istar(pep, µ)
    val, λ, k = alt_min(pep, w, µ)
    #Return: distance to closest alternative, best arm and closest λ for that alternative, best arm and vector for closest point in model
    val, (k, λ), (star, μ);
end

function glrt(pep::LinearBestArm, P::Matrix, μ)
    #println("glrt matrix")
    # P is a matrix of size nb_answers*nb_arms (both of which may be different from length(µ)).
    # In the BAI case, nb_answers = nb_arms != length(µ)
    nb_answers = size(P)[1]
    @assert length(size(μ)) == 1
    val = 0
    λs = zeros(nb_answers, length(µ))  # nb_answers * length(µ0)
    star = istar(pep, µ)
    for i in 1:nb_answers
        if i != star
            λs[i, :] = copy(µ)  # µ belongs to ¬i
        else
            val_i, λ_i, _ = alt_min(pep, P[i,:], µ)
            #println("i $i ; val_i $val_i ; λ_i $λ_i")
            λs[i, :] = λ_i
            val += val_i
        end
    end

    val, λs, (star, μ);
end


# oracle problem
function oracle(pep::LinearBestArm, μ)
    throw("Unimplemented")
end


# oracle problem for best μ in confidence interval
function optimistic_oracle(pep::LinearBestArm, hμ, N)
    throw("Unimplemented")
end


struct LinearThreshold
    expfam; # common exponential family
    arms;  # array of size K of arms in R^d
    τ;  # threshold
end

nanswers(pep::LinearThreshold, μ) = 2^length(µ);
narms(pep::LinearThreshold, µ) = length(pep.arms);
istar(pep::LinearThreshold, μ) = 1 + sum([(µ[k] > pep.τ) * 2^(k-1) for k in 1:length(µ)]);
getexpfam(pep::LinearThreshold, k) = pep.expfam;

function alt_min(pep::LinearThreshold, w, µ, k)
    #TODO: this functions assumes that expfam is Gaussian(1)
    #println("alt_min for k=$k, w=$w and µ=$µ")
    sum_w = sum(w)
    w = w/sum(w)  # avoid dividing by small quantities
    #println("w : $w")
    K = narms(pep, µ)
    # Construction of the matrix V = (sum_k w_k a_k a_k^T)^{-1}
    sum_arms_matrix = zeros(length(µ),length(µ))
    for j in 1:K
        sum_arms_matrix .+= w[j].* (pep.arms[j]*transpose(pep.arms[j]))
    end
    Vinv = inv(sum_arms_matrix)
    #println("Vinv=$Vinv")
    # Closest point
    η = (µ[k] - pep.τ) / Vinv[k,k]
    #println("η $η; sum_w $sum_w")
    λ = µ .- η * Vinv[:,k]
    # Divergence to that point
    val = .5 * sum_w * (µ[k] - pep.τ)^2 / Vinv[k,k]
    #println("k $k; val $val; λ $λ")
    return val, λ, k
end

function alt_min(pep::LinearThreshold, w, µ)
    minimum(alt_min(pep, w, µ, i) for i in 1:length(µ))
end

function glrt(pep::LinearThreshold, w::Vector, μ)
    #println("glrt vector")
    @assert length(size(μ)) == 1
    star = istar(pep, µ)
    val, λ, k = alt_min(pep, w, µ)
    answer = star
    if µ[k]>pep.τ
        answer -= 2^(k-1)
    else
        answer += 2^(k-1)
    end
    #Return: distance to closest alternative, answer and closest λ for that alternative, answer and vector for closest point in model
    val, (answer, λ), (star, μ);
end

function glrt(pep::LinearThreshold, P::Matrix, μ)
    #println("glrt matrix")
    # P is a matrix of size nb_answers*nb_arms (both of which may be different from length(µ)).
    # In the BAI case, nb_answers = nb_arms != length(µ)
    nb_answers = nanswers(pep, µ)
    @assert length(size(μ)) == 1
    val = 0
    λs = zeros(nb_answers, length(µ))
    star = istar(pep, µ)
    for i in 1:nb_answers
        if i != star
            λs[i, :] = copy(µ)  # µ belongs to ¬i
        else
            val_i, λ_i, _ = alt_min(pep, P[i,:], µ)
            #println("i $i ; val_i $val_i ; λ_i $λ_i")
            λs[i, :] = λ_i
            val += val_i
        end
    end

    val, λs, (star, μ);
end


# oracle problem
function oracle(pep::LinearThreshold, μ)
    throw("Unimplemented")
end


# oracle problem for best μ in confidence interval
function optimistic_oracle(pep::LinearThreshold, hμ, N)
    throw("Unimplemented")
end

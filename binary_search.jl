# Binary search for zero of continuous function f
# with one zero-crossing in [lo, hi]
function binary_search(f, lo, hi; ϵ = 1e-10, maxiter = 100)
    if f(lo) > 0
        @warn "f($(lo))=$(f(lo)) should be negative at low end";
        return lo;
    end
    if f(hi) < 0
        @warn "f($(hi))=$(f(hi)) should be positive at high end";
        return hi;
    end

    # @assert f(lo) < +ϵ "f($(lo))=$(f(lo)) should be negative at low end";
    # @assert f(hi) > -ϵ "f($(hi))=$(f(hi)) should be positive at high end";

    for i in 1:maxiter
        mid = (lo+hi)/2;
        if mid == lo || mid == hi
            # not going to get any better ...
            #@warn "binary_search collapsed after $i iterations.\nf($(lo)) = $(f(lo))\nf($(hi)) = $(f(hi))";
            return mid;
        end
        fmid = f(mid);
        if fmid < -ϵ
            lo = mid
        elseif fmid > ϵ
            hi = mid;
        else
            return mid;
        end
    end
    @warn "binary_search did not reach tolerance $ϵ in $maxiter iterations.\nf($(lo)) = $(f(lo))\nf($(hi)) = $(f(hi)),\nmid would be $((lo+hi)/2)";
    return (lo+hi)/2;
end

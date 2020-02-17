function solve(pep, μ, δ, β)
    Tstar, wstar = oracle(pep, μ);
    ⋆ = istar(pep, μ);

    # lower bound
    kl = (1-2δ)*log((1-δ)/δ);
    lbd = Tstar*kl;

    # more practical lower bound with the employed threshold β
    practical = binary_search(t -> t-Tstar*β(t), max(1, lbd), 1e10);

    Tstar, wstar, ⋆, lbd, practical;
end


function dump_stats(pep, μ, δs, βs, srs, datas)

    for i in 1:length(δs)
        δ = δs[i];
        β = βs[i];
        data = getindex.(datas, i);

        Tstar, wstar, ⋆, lbd, practical = solve(pep, μ, δ, β)

        rule = repeat("-", 60);

        println("");
        println(rule);
        println("$pep at δ = $δ");
        println(@sprintf("%-30s", "Arm"),
                join(map(k -> @sprintf("%6s", k), 1:length(μ))), " ",
                @sprintf("%7s", "total"), "  ",
                @sprintf("%7s", "err"), "  ",
                @sprintf("%7s", "time"));
        println(@sprintf("%-30s", "μ"),
                join(map(x -> @sprintf("%6.2f", x), μ)));
        println(@sprintf("%-30s", "w⋆"),
                join(map(x -> @sprintf("%6.2f", x), wstar)));
        println(rule);
        println(@sprintf("%-30s", "oracle"),
                join(map(w -> @sprintf("%6.0f", lbd*w), wstar)), " ",
                @sprintf("%7.0f", lbd));
        println(@sprintf("%-30s", "practical"),
                join(map(w -> @sprintf("%6.0f", practical*w), wstar)), " ",
                @sprintf("%7.0f", practical));
        println(rule);

        for r in eachindex(srs)
            Eτ = sum(x->sum(x[2]), data[r,:])/N;
            err = sum(x->x[1].!=⋆, data[r,:])/N;
            tim = sum(x->x[3],     data[r,:])/N;
            println(@sprintf("%-30s", long(srs[r])),
                    join(map(k -> @sprintf("%6.0f", sum(x->x[2][k], data[r,:])/N), eachindex(μ))), " ",
                    @sprintf("%7.0f", Eτ), "  ",
                    @sprintf("%7.5f", err), "  ",
                    @sprintf("%7.5f", tim/1e6)
                    );
            if err > δ
                @warn "too many errors for $(srs[r])";
            end
        end
        println(rule);
    end
end


function dump_stats(pep, μ, δs, βs, srs, datas, repeats)
    K = narms(pep, µ)

    for i in 1:length(δs)
        δ = δs[i];
        β = βs[i];
        data = getindex.(datas, i);

        #Tstar, wstar, ⋆, lbd, practical = solve(pep, μ, δ, β)
        ⋆ = istar(pep, µ)

        rule = repeat("-", 60);

        println("");
        println(rule);
        println("$pep at δ = $δ");
        println(@sprintf("%-30s", "Arm"),
                join(map(k -> @sprintf("%6s", k), 1:K)), " ",
                @sprintf("%7s", "total"), "  ",
                @sprintf("%7s", "err"), "  ",
                @sprintf("%7s", "time"));
        println(@sprintf("%-30s", "a\'μ"),
                join(map(x -> @sprintf("%6.2f", x'µ), pep.arms)));
        #println(@sprintf("%-30s", "w⋆"),
        #        join(map(x -> @sprintf("%6.2f", x), wstar)));
        println(rule);
        #println(@sprintf("%-30s", "oracle"),
        #        join(map(w -> @sprintf("%6.0f", lbd*w), wstar)), " ",
        #        @sprintf("%7.0f", lbd));
        #println(@sprintf("%-30s", "practical"),
        #        join(map(w -> @sprintf("%6.0f", practical*w), wstar)), " ",
        #        @sprintf("%7.0f", practical));
        println(rule);

        for r in eachindex(srs)
            Eτ = sum(x->sum(x[2]), data[r,:])/repeats;
            err = sum(x->x[1].!=⋆, data[r,:])/repeats;
            tim = sum(x->x[4],     data[r,:])/repeats;
            println(@sprintf("%-30s", long(srs[r])),
                    join(map(k -> @sprintf("%6.0f", sum(x->x[2][k], data[r,:])/repeats), 1:K)), " ",
                    @sprintf("%7.0f", Eτ), "  ",
                    @sprintf("%7.5f", err), "  ",
                    @sprintf("%7.5f", tim/1e6)
                    );
            if err > δ
                @warn "too many errors for $(srs[r])";
            end
        end
        println(rule);
    end
end


function τhist(pep, μ, δ, β, srs, data)
    Tstar, wstar, ⋆, lbd, practical = solve(pep, μ, δ, β)

    stephist(map(x -> sum(x[2]), data)', label=permutedims(collect(abbrev.(srs))));
    vline!([lbd], label="lower bd");
    vline!([practical], label="practical");
end


function boxes(pep, μ, δ, β, srs, data)
    #Tstar, wstar, ⋆, lbd, practical = solve(pep, μ, δ, β)

    xs = permutedims(collect(abbrev.(srs)));

    means = sum(sum.(getindex.(data,2)),dims=2)/repeats;

    #plot([lbd practical], seriestype=:hline, label=["lower bd" "practical"], legend=:top)
    plot(legend=:top)

    if true
        # violin(
        #     xs,
        #     map(x -> sum(x[2]), data)',
        #     label=""
        # )
        boxplot!(
            xs,
            map(x -> sum(x[2]), data)',
            label="",
            notch=false,
            outliers=true)

        plot!(xs,
              means', marker=(:star4,10,:black), label="");
        #top = max(4*practical, 2*maximum(means));
        # hard coded xlims are a dirty hack; I cannot figure out how
        # to draw the first two hlines without upsetting the x-axis
        #plot!(xlims=(1.2,15.4), ylims=(-.025*top, top));
    else
        bar(collect(abbrev.(srs)),
            means,
            label="");
    end

end


function randmax(vector, rank = 1)
   # returns an integer, not a CartesianIndex
    vector = vec(vector)
    Sorted = sort(vector, rev = true)
    m = Sorted[rank]
    Ind = findall(x -> x == m, vector)
    index = Ind[floor(Int, length(Ind) * rand())+1]
    return index
end


function randmin(vector, rank = 1)
   # returns an integer, not a CartesianIndex
    vector = vec(vector)
    Sorted = sort(vector, rev = false)
    m = Sorted[rank]
    Ind = findall(x -> x == m, vector)
    index = Ind[floor(Int, length(Ind) * rand())+1]
    return index
end

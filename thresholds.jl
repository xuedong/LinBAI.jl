# Some thresholds, implemented as types so we can save them with JLD2
# (which does not allow saving functions directly)


# Threshold from Optimal Best Arm Identification with Fixed Confidence
# (Garivier and Kaufmann 2016). Recommended in section 6.
struct GK16
    δ;
end

function (β::GK16)(t)
    log((log(t)+1)/β.δ)
end

function kl_divergence(P::Vector{Float64}, Q::Vector{Float64})
    return sum(P .* log.(P ./ Q)) # nats
end

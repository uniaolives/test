# modules/julia/src/hyperprompt_optimizer.jl

module HyperpromptOptimizer

using Flux
using Statistics
using LinearAlgebra

export optimize_hyperprompt, compute_free_energy

"""
    compute_free_energy(q_llm, q_human, p, beta)

Calcula F = KL(q_llm || p) + KL(q_human || p) + beta * KL(q_llm || q_human)
"""
function compute_free_energy(q_llm, q_human, p, beta)
    kl(q, p) = sum(q .* log.(q ./ p))
    return kl(q_llm, p) + kl(q_human, p) + beta * kl(q_llm, q_human)
end

"""
    optimize_hyperprompt(initial_vec, p_prior, beta, n_iter)

Otimiza o hiperprompt (distribuição q) para minimizar a energia livre variacional.
Usa Flux.jl para gradiente real.
"""
function optimize_hyperprompt(initial_vec, p_prior, beta, n_iter; lr=0.01)
    q = Flux.params(copy(initial_vec))

    # Função de perda: Energia Livre Variacional
    # F = D_KL[q || p] + beta * D_KL[q || q_human_mock]
    # Usamos um mock fixo para q_human baseado no prior para simplificação
    q_human = copy(p_prior)

    function loss(q_val)
        # Garantir que q é uma distribuição (softmax)
        q_soft = softmax(q_val)
        kl(q_dist, p_dist) = sum(q_dist .* log.(q_dist ./ (p_dist .+ 1e-9) .+ 1e-9))
        return kl(q_soft, p_prior) + beta * kl(q_soft, q_human)
    end

    opt = Flux.Adam(lr)

    for i in 1:n_iter
        grads = Flux.gradient(() -> loss(q[1]), q)
        Flux.update!(opt, q, grads)
    end

    return softmax(q[1])
end

end # module

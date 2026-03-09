// arkhe-core/cpp/include/vessel/pyg_bridge.hpp
// Expose OmegaFusion to C++ Arkhe(n) kernel using LibTorch

#ifndef ARKHE_PYG_BRIDGE_HPP
#define ARKHE_PYG_BRIDGE_HPP

#include <torch/script.h>  // LibTorch
#include <iostream>
#include <string>
#include <vector>

namespace Arkhe {
namespace Vessel {

struct FusionResult {
    double phi_q;
    bool channel_open;
    at::Tensor temporal_sig;
    at::Tensor zk_proofs;
};

class PyGBridge {
    torch::jit::script::Module omega_module_;
    double miller_limit_;

public:
    PyGBridge(const std::string& model_path, double miller_limit = 4.64)
        : miller_limit_(miller_limit) {
        try {
            omega_module_ = torch::jit::load(model_path);
            std::cout << "[PyGBridge] Model loaded successfully from " << model_path << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "[PyGBridge] Error loading the model: " << e.msg() << std::endl;
        }
    }

    FusionResult merge(const torch::Tensor& x,
                      const torch::Tensor& edge_index,
                      const torch::Tensor& edge_attr,
                      const torch::Tensor& batch) {

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(x);
        inputs.push_back(edge_index);
        inputs.push_back(edge_attr);
        inputs.push_back(batch);

        auto outputs = omega_module_.forward(inputs).toTuple();

        // Output indices based on OmegaFusion.forward return:
        // return phi_q, temporal_sig, zk_proofs, global_state
        at::Tensor phi_q_tensor = outputs->elements()[0].toTensor();
        at::Tensor temporal_sig = outputs->elements()[1].toTensor();
        at::Tensor zk_proofs = outputs->elements()[2].toTensor();

        double phi_q_val = phi_q_tensor.item<double>();

        return FusionResult{
            .phi_q = phi_q_val,
            .channel_open = (phi_q_val > miller_limit_),
            .temporal_sig = temporal_sig,
            .zk_proofs = zk_proofs
        };
    }
};

} // namespace Vessel
} // namespace Arkhe

#endif // ARKHE_PYG_BRIDGE_HPP

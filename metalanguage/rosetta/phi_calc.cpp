#include <vector>
#include <numeric>

float calculate_integration_phi(const std::vector<float>& agent_a, const std::vector<float>& agent_b) {
    return std::inner_product(agent_a.begin(), agent_a.end(), agent_b.begin(), 0.0f);
}

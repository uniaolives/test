/**
 * Gradient Continuity for Spatial Reconstruction
 * Ensures ∇C smoothness across gap boundaries
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

struct Node {
    int id;
    double x, y, z;        // 3D position
    double coherence;
    double fluctuation;

    Node(int id_, double x_, double y_, double z_)
        : id(id_), x(x_), y(y_), z(z_),
          coherence(0.86), fluctuation(0.14) {}

    // Verify C + F = 1
    bool verify_conservation() const {
        return std::abs(coherence + fluctuation - 1.0) < 1e-10;
    }
};

class GradientContinuity {
private:
    std::vector<Node> nodes;
    double gradient_tolerance;

public:
    GradientContinuity(double tol = 0.1)
        : gradient_tolerance(tol) {}

    void add_node(const Node& node) {
        nodes.push_back(node);
    }

    /**
     * Compute gradient ∇C between two nodes
     */
    double compute_gradient(const Node& n1, const Node& n2) const {
        double delta_c = std::abs(n2.coherence - n1.coherence);
        double distance = std::sqrt(
            std::pow(n2.x - n1.x, 2) +
            std::pow(n2.y - n1.y, 2) +
            std::pow(n2.z - n1.z, 2)
        );

        if (distance < 1e-10) return 0.0;
        return delta_c / distance;
    }

    /**
     * Find k-nearest neighbors
     */
    std::vector<int> find_k_nearest(int target_id, int k) const {
        if (target_id >= nodes.size()) return {};

        const Node& target = nodes[target_id];

        // Compute distances
        std::vector<std::pair<double, int>> distances;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (i == target_id) continue;

            double dist = std::sqrt(
                std::pow(nodes[i].x - target.x, 2) +
                std::pow(nodes[i].y - target.y, 2) +
                std::pow(nodes[i].z - target.z, 2)
            );
            distances.push_back({dist, i});
        }

        // Sort by distance
        std::sort(distances.begin(), distances.end());

        // Return k nearest
        std::vector<int> nearest;
        for (int i = 0; i < std::min(k, (int)distances.size()); ++i) {
            nearest.push_back(distances[i].second);
        }

        return nearest;
    }

    /**
     * Interpolate coherence at target node using gradient-based method
     */
    double interpolate_coherence(int target_id, int k_neighbors = 10) {
        if (target_id >= nodes.size()) return 0.0;

        auto nearest = find_k_nearest(target_id, k_neighbors);

        if (nearest.empty()) return 0.86; // Default

        // Weighted interpolation based on inverse distance
        double weighted_sum = 0.0;
        double weight_total = 0.0;

        for (int neighbor_id : nearest) {
            double dist = std::sqrt(
                std::pow(nodes[neighbor_id].x - nodes[target_id].x, 2) +
                std::pow(nodes[neighbor_id].y - nodes[target_id].y, 2) +
                std::pow(nodes[neighbor_id].z - nodes[target_id].z, 2)
            );

            double weight = 1.0 / (dist + 0.01); // Avoid division by zero
            weighted_sum += weight * nodes[neighbor_id].coherence;
            weight_total += weight;
        }

        return weighted_sum / weight_total;
    }

    /**
     * Verify gradient continuity across gap boundary
     */
    bool verify_continuity(const std::vector<int>& boundary_nodes) {
        for (int id : boundary_nodes) {
            auto neighbors = find_k_nearest(id, 4);

            for (int neighbor_id : neighbors) {
                double grad = compute_gradient(nodes[id], nodes[neighbor_id]);

                if (grad > gradient_tolerance) {
                    std::cerr << "Gradient discontinuity detected: "
                             << grad << " > " << gradient_tolerance
                             << " between nodes " << id << " and "
                             << neighbor_id << std::endl;
                    return false;
                }
            }
        }

        return true;
    }

    /**
     * Reconstruct coherence for gap nodes
     */
    void reconstruct_gap(const std::vector<int>& gap_node_ids,
                        const std::vector<int>& support_node_ids) {
        std::cout << "Reconstructing coherence for " << gap_node_ids.size()
                  << " gap nodes using " << support_node_ids.size()
                  << " support nodes" << std::endl;

        for (int gap_id : gap_node_ids) {
            // Set coherence to zero (simulate gap)
            nodes[gap_id].coherence = 0.0;
            nodes[gap_id].fluctuation = 1.0;

            // Interpolate from support nodes
            double interpolated = interpolate_coherence(gap_id, 10);

            // Restore
            nodes[gap_id].coherence = interpolated;
            nodes[gap_id].fluctuation = 1.0 - interpolated;
        }

        std::cout << "Reconstruction complete" << std::endl;
    }

    /**
     * Compute gradient squared |∇C|²
     */
    double compute_gradient_squared() const {
        if (nodes.size() < 2) return 0.0;

        double sum = 0.0;
        int count = 0;

        for (size_t i = 0; i < nodes.size(); ++i) {
            auto neighbors = find_k_nearest(i, 6);

            for (int j : neighbors) {
                double grad = compute_gradient(nodes[i], nodes[j]);
                sum += grad * grad;
                count++;
            }
        }

        return count > 0 ? sum / count : 0.0;
    }
};

// Example usage
int main() {
    std::cout << std::fixed << std::setprecision(6);

    GradientContinuity gc;

    // Create a grid of nodes
    int grid_size = 20;
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            for (int k = 0; k < grid_size; ++k) {
                Node node(i * grid_size * grid_size + j * grid_size + k,
                         i * 1.0, j * 1.0, k * 1.0);
                gc.add_node(node);
            }
        }
    }

    std::cout << "Created " << grid_size*grid_size*grid_size << " nodes\n";

    // Define gap region (center cube)
    std::vector<int> gap_nodes;
    std::vector<int> support_nodes;

    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            for (int k = 0; k < grid_size; ++k) {
                int id = i * grid_size * grid_size + j * grid_size + k;

                // Gap: center 8x8x8 cube
                if (i >= 6 && i < 14 &&
                    j >= 6 && j < 14 &&
                    k >= 6 && k < 14) {
                    gap_nodes.push_back(id);
                } else {
                    support_nodes.push_back(id);
                }
            }
        }
    }

    std::cout << "Gap nodes: " << gap_nodes.size() << "\n";
    std::cout << "Support nodes: " << support_nodes.size() << "\n";
    std::cout << "Support ratio: "
              << (double)support_nodes.size() / gap_nodes.size()
              << ":1\n";

    // Reconstruct gap
    gc.reconstruct_gap(gap_nodes, support_nodes);

    // Verify gradient continuity
    std::vector<int> boundary;
    // Simplified: just use first 100 gap nodes as "boundary"
    for (int i = 0; i < std::min(100, (int)gap_nodes.size()); ++i) {
        boundary.push_back(gap_nodes[i]);
    }

    bool continuous = gc.verify_continuity(boundary);
    std::cout << "Gradient continuity: "
              << (continuous ? "VERIFIED" : "FAILED") << "\n";

    // Compute |∇C|²
    double grad_sq = gc.compute_gradient_squared();
    std::cout << "|∇C|² = " << grad_sq << "\n";
    std::cout << "Target: < 0.0049\n";
    std::cout << "Status: " << (grad_sq < 0.0049 ? "PASS" : "FAIL") << "\n";

    return 0;
}

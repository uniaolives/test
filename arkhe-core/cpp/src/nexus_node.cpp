#include "nexus_node.hpp"
#include <algorithm>

namespace arkhe::nexus {

NexusNode::NexusNode(const NexusID& id, const NexusEndpoint& ep, double initial_coherence)
    : id_(id), endpoint_(ep), coherence_(initial_coherence), exergy_buffer_(initial_coherence * 1000), last_seen_(0) {}

bool NexusNode::add_neighbor(const NexusNode& neighbor) {
    auto it = std::find_if(neighbors_.begin(), neighbors_.end(),
        [&](const NexusID& n) { return n.id == neighbor.get_id().id; });
    if (it != neighbors_.end()) return false;
    neighbors_.push_back(neighbor.get_id());
    return true;
}

bool NexusNode::deduct_exergy(double amount) {
    if (exergy_buffer_ < amount) return false;
    exergy_buffer_ -= amount;
    return true;
}

void NexusNode::add_exergy(double amount) {
    exergy_buffer_ += amount;
}

std::optional<RoutingEntry> NexusNode::lookup_route(const NexusID& destination) const {
    auto it = routing_table_.find(destination.id);
    if (it != routing_table_.end()) return it->second;
    return std::nullopt;
}

bool NexusNode::remove_neighbor(const NexusID& neighbor_id) {
    auto it = std::find_if(neighbors_.begin(), neighbors_.end(),
        [&](const NexusID& n) { return n.id == neighbor_id.id; });
    if (it == neighbors_.end()) return false;
    neighbors_.erase(it);
    return true;
}

std::vector<NexusID> NexusNode::get_neighbors() const {
    return neighbors_;
}

bool NexusNode::add_route(const RoutingEntry& route) {
    routing_table_[route.destination.id] = route;
    return true;
}

std::vector<uint8_t> NexusNode::serialize() const {
    // Mock serialization
    return std::vector<uint8_t>();
}

NexusNode NexusNode::deserialize(const std::vector<uint8_t>& data) {
    // Mock deserialization
    return NexusNode({0, ""}, {"", 0}, 0.0);
}

} // namespace arkhe::nexus

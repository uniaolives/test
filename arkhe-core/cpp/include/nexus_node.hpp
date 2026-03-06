#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <optional>

namespace arkhe {
namespace nexus {

struct NexusID {
    uint64_t id;
    std::string human_readable;  // ex: "2009-01-03-Satoshi"
};

struct NexusEndpoint {
    std::string address;  // IP ou identificador de rede
    uint16_t port;
};

struct RoutingEntry {
    NexusID destination;
    NexusID next_hop;
    double coherence_cost;   // custo acumulado esperado
    uint64_t last_updated;
};

class NexusNode {
public:
    NexusNode(const NexusID& id, const NexusEndpoint& ep, double initial_coherence);

    // Getters
    NexusID get_id() const { return id_; }
    double get_coherence() const { return coherence_; }
    double get_exergy() const { return exergy_buffer_; }

    // Gestão de vizinhos
    bool add_neighbor(const NexusNode& neighbor);
    bool remove_neighbor(const NexusID& neighbor_id);
    std::vector<NexusID> get_neighbors() const;

    // Roteamento
    bool add_route(const RoutingEntry& route);
    std::optional<RoutingEntry> lookup_route(const NexusID& destination) const;

    // Handover
    bool deduct_exergy(double amount);   // pagar quantum interest
    void add_exergy(double amount);      // receber coerência

    // Serialização para ledger
    std::vector<uint8_t> serialize() const;
    static NexusNode deserialize(const std::vector<uint8_t>& data);

private:
    NexusID id_;
    NexusEndpoint endpoint_;
    double coherence_;        // coerência local (λ₂)
    double exergy_buffer_;    // reserva energética
    std::map<uint64_t, RoutingEntry> routing_table_;
    std::vector<NexusID> neighbors_;
    uint64_t last_seen_;
};
} // namespace nexus
} // namespace arkhe

// OrbEngine.hpp
#include <iostream>
#include <vector>
#include <cstdint>

class WormholeThroat {
public:
    double entrance_lat, entrance_lon;
    double exit_lat, exit_lon;
    double bandwidth;

    WormholeThroat(double b) : bandwidth(b) {}
};

class Orb {
private:
    double stability;
    WormholeThroat throat;

public:
    Orb(double lambda, double freq) : throat(freq) {
        stability = lambda;
    }

    bool transmit(std::vector<uint8_t>& handover) {
        if (stability > 0.5) {
            // A transmitir através do wormhole
            return true;
        }
        return false;
    }
};

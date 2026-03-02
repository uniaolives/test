#pragma once
#include <cstdint>

namespace arkhe {

template<uint64_t MOD>
struct mint {
    uint64_t val;
    mint(uint64_t v = 0) : val(v % MOD) {}
    mint operator+(mint m) const { return {(val + m.val) % MOD}; }
    mint operator-(mint m) const { return {(val + MOD - m.val) % MOD}; }
    mint operator*(mint m) const { return {(val * m.val) % MOD}; }
};

} // namespace arkhe

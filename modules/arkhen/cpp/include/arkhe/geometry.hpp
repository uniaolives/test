#pragma once
#include <cmath>
#include <vector>

namespace arkhe {

struct point {
    double x, y;
    point operator+(point p) const { return {x + p.x, y + p.y}; }
    point operator-(point p) const { return {x - p.x, y - p.y}; }
    point operator*(double d) const { return {x * d, y * d}; }
    double dot(point p) const { return x * p.x + y * p.y; }
    double cross(point p) const { return x * p.y - y * p.x; }
    double dist() const { return std::sqrt(x * x + y * y); }
};

inline double area(const std::vector<point>& poly) {
    double a = 0;
    for (size_t i = 0; i < poly.size(); ++i) {
        a += poly[i].cross(poly[(i + 1) % poly.size()]);
    }
    return std::abs(a) / 2.0;
}

} // namespace arkhe

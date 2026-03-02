// ucd.cpp – Universal Coherence Detection in C++
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

double pearson(const std::vector<double>& x, const std::vector<double>& y) {
    size_t n = x.size();
    double meanX = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double meanY = std::accumulate(y.begin(), y.end(), 0.0) / n;
    double num = 0.0, denX = 0.0, denY = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - meanX;
        double dy = y[i] - meanY;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
    }
    return (denX == 0 || denY == 0) ? 0 : num / std::sqrt(denX * denY);
}

int main() {
    std::vector<std::vector<double>> data = {{1, 2, 3, 4}, {2, 3, 4, 5}, {5, 6, 7, 8}};
    double sumCorr = 0;
    int count = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = i + 1; j < data.size(); ++j) {
            sumCorr += std::abs(pearson(data[i], data[j]));
            count++;
        }
    }
    double C = count > 0 ? sumCorr / count : 0.5;
    double F = 1.0 - C;
    std::cout << "C: " << C << ", F: " << F << ", C+F: " << (C + F) << std::endl;
    return 0;
}

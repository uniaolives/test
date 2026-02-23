#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include <algorithm>

// RK4 constants for Aizawa
const double A = 0.95;
const double B = 0.7;
const double C = 0.6;
const double D = 3.5;
const double E = 0.25;
const double F = 0.1;

// Scalar version of RK4 step for remainder
void rk4_step_scalar(double& x, double& y, double& z, double dt, int steps) {
    for (int s = 0; s < steps; ++s) {
        auto derivatives = [&](double cx, double cy, double cz, double& dx, double& dy, double& dz) {
            dx = (cz - B) * cx - D * cy;
            dy = D * cx + (cz - B) * cy;
            dz = C + A * cz - (cz * cz * cz) / 3.0 - (cx * cx + cy * cy) * (1.0 + E * cz) + F * cz * (cx * cx * cx);
        };

        double k1x, k1y, k1z;
        derivatives(x, y, z, k1x, k1y, k1z);

        double k2x, k2y, k2z;
        derivatives(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z, k2x, k2y, k2z);

        double k3x, k3y, k3z;
        derivatives(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z, k3x, k3y, k3z);

        double k4x, k4y, k4z;
        derivatives(x + dt * k3x, y + dt * k3y, z + dt * k3z, k4x, k4y, k4z);

        x += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        y += (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
        z += (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
    }
}

void rk4_step_avx2_block(double* xs, double* ys, double* zs, double dt, int steps) {
    __m256d vdt = _mm256_set1_pd(dt);
    __m256d vdt_half = _mm256_set1_pd(dt * 0.5);
    __m256d vdt_sixth = _mm256_set1_pd(dt / 6.0);
    __m256d vtwo = _mm256_set1_pd(2.0);
    __m256d vone = _mm256_set1_pd(1.0);

    __m256d va = _mm256_set1_pd(A);
    __m256d vb = _mm256_set1_pd(B);
    __m256d vc = _mm256_set1_pd(C);
    __m256d vd = _mm256_set1_pd(D);
    __m256d ve = _mm256_set1_pd(E);
    __m256d vf = _mm256_set1_pd(F);
    __m256d vone_third = _mm256_set1_pd(1.0 / 3.0);

    __m256d x = _mm256_loadu_pd(xs);
    __m256d y = _mm256_loadu_pd(ys);
    __m256d z = _mm256_loadu_pd(zs);

    for (int s = 0; s < steps; ++s) {
        auto compute_derivatives = [&](__m256d cur_x, __m256d cur_y, __m256d cur_z,
                                      __m256d& dx, __m256d& dy, __m256d& dz) {
            __m256d z_minus_b = _mm256_sub_pd(cur_z, vb);
            dx = _mm256_sub_pd(_mm256_mul_pd(z_minus_b, cur_x), _mm256_mul_pd(vd, cur_y));
            dy = _mm256_add_pd(_mm256_mul_pd(vd, cur_x), _mm256_mul_pd(z_minus_b, cur_y));

            __m256d z2 = _mm256_mul_pd(cur_z, cur_z);
            __m256d z3 = _mm256_mul_pd(z2, cur_z);
            __m256d x2 = _mm256_mul_pd(cur_x, cur_x);
            __m256d y2 = _mm256_mul_pd(cur_y, cur_y);
            __m256d x3 = _mm256_mul_pd(x2, cur_x);

            __m256d term1 = _mm256_add_pd(vc, _mm256_mul_pd(va, cur_z));
            __m256d term2 = _mm256_mul_pd(z3, vone_third);
            __m256d xy2 = _mm256_add_pd(x2, y2);
            __m256d one_plus_ez = _mm256_add_pd(vone, _mm256_mul_pd(ve, cur_z));
            __m256d term3 = _mm256_mul_pd(xy2, one_plus_ez);
            __m256d term4 = _mm256_mul_pd(_mm256_mul_pd(vf, cur_z), x3);

            dz = _mm256_add_pd(_mm256_sub_pd(_mm256_sub_pd(term1, term2), term3), term4);
        };

        __m256d k1x, k1y, k1z;
        compute_derivatives(x, y, z, k1x, k1y, k1z);

        __m256d k2x, k2y, k2z;
        compute_derivatives(_mm256_add_pd(x, _mm256_mul_pd(vdt_half, k1x)),
                            _mm256_add_pd(y, _mm256_mul_pd(vdt_half, k1y)),
                            _mm256_add_pd(z, _mm256_mul_pd(vdt_half, k1z)),
                            k2x, k2y, k2z);

        __m256d k3x, k3y, k3z;
        compute_derivatives(_mm256_add_pd(x, _mm256_mul_pd(vdt_half, k2x)),
                            _mm256_add_pd(y, _mm256_mul_pd(vdt_half, k2y)),
                            _mm256_add_pd(z, _mm256_mul_pd(vdt_half, k2z)),
                            k3x, k3y, k3z);

        __m256d k4x, k4y, k4z;
        compute_derivatives(_mm256_add_pd(x, _mm256_mul_pd(vdt, k3x)),
                            _mm256_add_pd(y, _mm256_mul_pd(vdt, k3y)),
                            _mm256_add_pd(z, _mm256_mul_pd(vdt, k3z)),
                            k4x, k4y, k4z);

        x = _mm256_add_pd(x, _mm256_mul_pd(vdt_sixth, _mm256_add_pd(_mm256_add_pd(k1x, _mm256_mul_pd(vtwo, k2x)), _mm256_add_pd(_mm256_mul_pd(vtwo, k3x), k4x))));
        y = _mm256_add_pd(y, _mm256_mul_pd(vdt_sixth, _mm256_add_pd(_mm256_add_pd(k1y, _mm256_mul_pd(vtwo, k2y)), _mm256_add_pd(_mm256_mul_pd(vtwo, k3y), k4y))));
        z = _mm256_add_pd(z, _mm256_mul_pd(vdt_sixth, _mm256_add_pd(_mm256_add_pd(k1z, _mm256_mul_pd(vtwo, k2z)), _mm256_add_pd(_mm256_mul_pd(vtwo, k3z), k4z))));
    }

    _mm256_storeu_pd(xs, x);
    _mm256_storeu_pd(ys, y);
    _mm256_storeu_pd(zs, z);
}

int main() {
    const int n = 1000000;
    const int steps = 100;
    const double dt = 0.01;

    std::vector<double> xs(n), ys(n), zs(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < n; ++i) {
        xs[i] = dis(gen);
        ys[i] = dis(gen);
        zs[i] = dis(gen);
    }

    std::cout << "Starting Aizawa C++ AVX2 + OpenMP Simulation with " << n << " points for " << steps << " steps..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i += 4) {
        if (i + 4 <= n) {
            rk4_step_avx2_block(&xs[i], &ys[i], &zs[i], dt, steps);
        } else {
            for (int j = i; j < n; ++j) {
                rk4_step_scalar(xs[j], ys[j], zs[j], dt, steps);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Simulation completed in " << diff.count() << " seconds." << std::endl;
    std::cout << "Performance: " << (double)n * steps / diff.count() / 1e6 << " million iterations per second." << std::endl;

    return 0;
}

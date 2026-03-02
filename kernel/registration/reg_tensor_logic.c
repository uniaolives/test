// kernel/registration/reg_tensor_logic.c
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/math.h>

/**
 * reg_tl_verify_nematic - Enforces the Nematic Order invariant (S ∈ [0.3, 0.95]).
 * @R: A pointer to the flattened d x d relation-embedding tensor.
 * @d: The dimension of the tensor.
 * @out_S: A pointer to a float where the computed S value will be stored.
 *
 * This function calculates the nematic scalar S = (1/d) * tr(R^T * R) and
 * verifies that it lies within the bounds required for a stable yet adaptable
 * "liquid crystal" phase. This check is critical for preventing model collapse
 * (overfitting) and divergence (underfitting).
 *
 * Returns:
 *  0 if the invariant holds.
 *  -EINVAL if the invariant is violated.
 */
int reg_tl_verify_nematic(const float* R, size_t d, float* out_S) {
    float trace = 0.0f;
    size_t i, j;

    if (!R || d == 0) {
        return -EINVAL;
    }

    // Calculate trace of R^T * R, which is the sum of squares of all elements
    for (i = 0; i < d * d; ++i) {
        trace += R[i] * R[i];
    }

    float S = trace / (float)d;
    *out_S = S;

    // Enforce the nematic phase bounds
    if (S < 0.3f || S > 0.95f) {
        pr_warn("TIM-TENSOR: Nematic invariant violated. S = %.4f\n", S);
        return -EINVAL;
    }

    return 0; // OK
}
EXPORT_SYMBOL_GPL(reg_tl_verify_nematic);

/**
 * reg_tl_verify_birkhoff - Enforces the doubly-stochastic invariant.
 * @A: A pointer to the flattened d x d attention-style matrix.
 * @d: The dimension of the matrix.
 * @epsilon: The tolerance for checking if row/column sums are equal to 1.
 *
 * This function verifies that the matrix A is doubly-stochastic, meaning that
 * the sum of elements in each row and each column is 1 (within a given tolerance).
 * This is a key property of attention mechanisms and other permutation-invariant
 * operators.
 *
 * Returns:
 *  0 if the invariant holds.
 *  -EINVAL if the invariant is violated.
 */
int reg_tl_verify_birkhoff(const float* A, size_t d, float epsilon) {
    size_t i, j;
    float row_sum, col_sum;

    if (!A || d == 0) {
        return -EINVAL;
    }

    for (i = 0; i < d; ++i) {
        row_sum = 0.0f;
        col_sum = 0.0f;
        for (j = 0; j < d; ++j) {
            row_sum += A[i * d + j];
            col_sum += A[j * d + i];
        }

        if (fabs(row_sum - 1.0f) > epsilon || fabs(col_sum - 1.0f) > epsilon) {
            pr_warn("TIM-TENSOR: Birkhoff invariant violated at row/col %zu. Row sum: %.4f, Col sum: %.4f\n",
                    i, row_sum, col_sum);
            return -EINVAL;
        }
    }

    return 0; // OK
}
EXPORT_SYMBOL_GPL(reg_tl_verify_birkhoff);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("TIM VM Tensor-Logic Invariant Guards");
MODULE_AUTHOR("TIM Architecture Group");

/* tests/test_tensor_invariants.c
 * Kernel module to unit-test the Tensor-Logic invariant guards.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include "../kernel/registration/reg_shared.h"

// --- Test Cases ---

static void test_nematic_guard(void)
{
    float *tensor;
    float S;
    int ret;
    size_t d = 4;

    pr_info("TIM-TEST: --- Running Nematic Guard Unit Tests ---\n");

    // 1. Test "Good" Tensor (S should be in [0.3, 0.95])
    // A mostly diagonal matrix should have a reasonable S value.
    tensor = kcalloc(d * d, sizeof(float), GFP_KERNEL);
    tensor[0] = 0.8f; tensor[5] = 0.7f; tensor[10] = 0.9f; tensor[15] = 0.85f;
    ret = reg_tl_verify_nematic(tensor, d, &S);
    pr_info("TIM-TEST: Nematic 'Good' Tensor (S=%.4f): %s\n", S, ret == 0 ? "PASS" : "FAIL");
    kfree(tensor);

    // 2. Test "Crystalline" Tensor (S > 0.95)
    // A rank-1 matrix, e.g., outer product of a vector with itself.
    tensor = kcalloc(d * d, sizeof(float), GFP_KERNEL);
    // Let v = [1, 1, 1, 1], then R = v' * v. Sum of squares = 16. S = 16 / 4 = 4.
    for(int i=0; i < d*d; i++) tensor[i] = 1.0f;
    ret = reg_tl_verify_nematic(tensor, d, &S);
    pr_info("TIM-TEST: Nematic 'Crystalline' Tensor (S=%.4f): %s\n", S, ret != 0 ? "PASS" : "FAIL");
    kfree(tensor);

    // 3. Test "Gaseous" Tensor (S < 0.3)
    // A very sparse or near-zero matrix.
    tensor = kcalloc(d * d, sizeof(float), GFP_KERNEL);
    tensor[0] = 0.1f; // Sum of squares = 0.01. S = 0.01 / 4 = 0.0025
    ret = reg_tl_verify_nematic(tensor, d, &S);
    pr_info("TIM-TEST: Nematic 'Gaseous' Tensor (S=%.4f): %s\n", S, ret != 0 ? "PASS" : "FAIL");
    kfree(tensor);
}

static void test_birkhoff_guard(void)
{
    float *matrix;
    int ret;
    size_t d = 3;
    float epsilon = 1e-6f;

    pr_info("TIM-TEST: --- Running Birkhoff Guard Unit Tests ---\n");

    // 1. Test "Good" Matrix (Doubly-Stochastic)
    // A permutation matrix is a perfect example.
    matrix = kcalloc(d * d, sizeof(float), GFP_KERNEL);
    matrix[1] = 1.0f; // 0,1
    matrix[3] = 1.0f; // 1,0
    matrix[8] = 1.0f; // 2,2
    ret = reg_tl_verify_birkhoff(matrix, d, epsilon);
    pr_info("TIM-TEST: Birkhoff 'Good' Matrix: %s\n", ret == 0 ? "PASS" : "FAIL");
    kfree(matrix);

    // 2. Test "Bad" Matrix (Row sum != 1)
    matrix = kcalloc(d * d, sizeof(float), GFP_KERNEL);
    matrix[0] = 0.5f; matrix[1] = 0.5f; // Row 0 sum = 1
    matrix[3] = 1.0f; // Row 1 sum = 1
    matrix[6] = 1.1f; // Row 2 sum = 1.1 (bad)
    ret = reg_tl_verify_birkhoff(matrix, d, epsilon);
    pr_info("TIM-TEST: Birkhoff 'Bad Row' Matrix: %s\n", ret != 0 ? "PASS" : "FAIL");
    kfree(matrix);

    // 3. Test "Bad" Matrix (Col sum != 1)
    matrix = kcalloc(d * d, sizeof(float), GFP_KERNEL);
    matrix[0] = 0.9f; // Col 0 has 0.9
    matrix[4] = 1.0f; // Col 1 has 1.0
    matrix[8] = 1.0f; // Col 2 has 1.0
    ret = reg_tl_verify_birkhoff(matrix, d, epsilon);
    pr_info("TIM-TEST: Birkhoff 'Bad Col' Matrix: %s\n", ret != 0 ? "PASS" : "FAIL");
    kfree(matrix);
}


static int __init test_invariants_init(void)
{
    pr_info("TIM-TEST: Loading Tensor Invariants Test Module...\n");
    test_nematic_guard();
    test_birkhoff_guard();
    // The module does its job at load time and can then be removed.
    return -1; // Return -1 to prevent the module from staying loaded.
}

static void __exit test_invariants_exit(void)
{
    // This will not be called due to the -1 return in init.
}

module_init(test_invariants_init);
module_exit(test_invariants_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Test module for TIM VM Tensor-Logic Invariants");
MODULE_AUTHOR("TIM Architecture Group");

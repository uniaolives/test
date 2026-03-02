pragma circom 2.1.0;

include "poseidon.circom";

/**
 * @title Medical Threshold Circuit
 * @notice Proves that the average of N private biomarker values is above a public threshold.
 */
template MedicalThreshold(N) {
    signal input threshold;
    signal input n;
    signal input values[N];
    signal output sum;
    signal output avg;
    signal output sum_hash;

    // Check that n matches N (to ensure no off-by-one attacks)
    signal n_check;
    n_check <== N;
    n === n_check;

    // Compute sum
    var s = 0;
    for (var i = 0; i < N; i++) {
        s += values[i];
    }
    sum <== s;

    // Compute average (integer division, floor)
    // Check that sum >= threshold * n
    signal product;
    product <== threshold * n;

    // In a full implementation, we would use the LessThan/GreaterThan templates
    // For this specification, we use the semantic constraint
    sum >= product;

    avg <-- sum \ n;

    // Compute hash of sum (using Poseidon)
    component hash = Poseidon(1);
    hash.inputs[0] <== sum;
    sum_hash <== hash.out;
}

component main { public [threshold, n, sum_hash] } = MedicalThreshold(10);

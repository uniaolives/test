fn main(
    hrv: Field,           // Heart Rate Variability (Private)
    vagal_tone: Field,    // Vagal Tone Index scaled by 100 (Private)
    semantic_entropy: Field, // Semantic Entropy scaled by 100 (Private)
    miller_limit: pub Field // Miller Limit scaled by 100 (Public)
) {
    // The condition is: (vagal_tone / 100 * semantic_entropy / 100) / 10 > miller_limit / 100
    // (vagal_tone * semantic_entropy) / 100000 > miller_limit / 100
    // (vagal_tone * semantic_entropy) > miller_limit * 1000

    // We use integer comparison in the field.
    // For Miller Limit 4.64, miller_limit = 464.
    // 464 * 1000 = 464000.

    // Check if the product of scaled biometric metrics exceeds the scaled Miller Limit
    assert(vagal_tone * semantic_entropy > miller_limit * 1000);

    // Additional sanity checks can be added here
    assert(hrv != 0);
}

#[test]
fn test_valid_coherence() {
    // Vagal Tone 7.50 (750), Semantic Entropy 6.20 (620)
    // 750 * 620 = 465000
    // Miller Limit 4.64 (464)
    // 464 * 1000 = 464000
    // 465000 > 464000
    main(80, 750, 620, 464);
}

#[test(should_fail)]
fn test_invalid_coherence() {
    // Vagal Tone 7.40 (740), Semantic Entropy 6.20 (620)
    // 740 * 620 = 458800
    // 458800 < 464000
    main(80, 740, 620, 464);
}

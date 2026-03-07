#[cfg(test)]
mod tests {
    use crate::physics::melonic_engine::*;
    use crate::physics::miller::PHI_Q;

    #[test]
    fn test_compute_f_extremum() {
        let n = 100;
        let coupling = 1.0;
        let f = compute_f_extremum(n, coupling);
        assert!(f >= PHI_Q);
        assert!(f < PHI_Q + 0.1);
    }

    #[test]
    fn test_is_melonic_dominant() {
        assert!(is_melonic_dominant(10, 5.0));
        assert!(!is_melonic_dominant(2, 5.0));
        assert!(!is_melonic_dominant(10, 4.0));
    }
}

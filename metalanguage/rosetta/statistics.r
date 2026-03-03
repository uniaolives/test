test_emergence <- function(perf_individual, perf_collective) {
    if(perf_collective > sum(perf_individual) * 1.5) {
        return("Hypothesis Supported")
    } else {
        return("Falsified")
    }
}

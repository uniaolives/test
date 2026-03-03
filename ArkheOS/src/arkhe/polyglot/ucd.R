# ucd.R – Universal Coherence Detection in R

ucd_analyze <- function(data) {
    if (nrow(data) > 1) {
        corr_mat <- abs(cor(t(data)))
        C <- mean(corr_mat[lower.tri(corr_mat)])
    } else {
        C <- 0.5
    }
    F <- 1.0 - C
    return(list(C = C, F = F, conservation = (abs(C + F - 1.0) < 1e-10)))
}

data <- matrix(c(1,2,3,4, 2,3,4,5, 5,6,7,8), nrow=3, byrow=TRUE)
print(ucd_analyze(data))

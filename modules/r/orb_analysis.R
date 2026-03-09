# orb_analysis.R

detect_orb <- function(rf_input, mesh_density) {
  stability <- (rf_input / 1e9) * mesh_density
  if (stability > 0.618) {
    return(list(stability=stability, status="ORB_DETECTED"))
  }
  return(list(stability=stability, status="NOISE"))
}

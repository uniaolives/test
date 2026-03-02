// build.rs – ativado com feature "fpga"
fn main() {
    println!("cargo:rerun-if-env-changed=FPGA_SYNTH");
    if std::env::var("FPGA_SYNTH").is_ok() {
        // Trigger synthesis placeholders
        println!("cargo:warning=FPGA synthesis triggered (placeholder)");
    }
}

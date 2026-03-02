fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .compile(
            &["proto/sasc_society.proto"],
            &["proto"],
        )?;
    Ok(())
}

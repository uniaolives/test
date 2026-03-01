use std::env;
use std::path::PathBuf;

fn main() {
    let kernel_build_dir = env::var("ARKHE_KERNEL_BUILD_DIR")
        .unwrap_or_else(|_| "../kernel/build".to_string());

    let path = PathBuf::from(&kernel_build_dir);
    if path.exists() {
        println!("cargo:rustc-link-search=native={}", path.display());
        println!("cargo:rustc-link-lib=dylib=arkhe_kernel");
    } else {
        println!("cargo:warning=Arkhe Kernel build directory not found at {}. FFI calls will fail at runtime unless libarkhe_kernel.so is in LD_LIBRARY_PATH.", path.display());
    }
}

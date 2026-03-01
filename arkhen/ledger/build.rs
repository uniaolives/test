fn main() {
    println!("cargo:rustc-link-search=native=../kernel/build");
    println!("cargo:rustc-link-lib=dylib=arkhe_kernel");
}

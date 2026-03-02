fn main() {
    tonic_build::compile_protos("../arkhed/proto/arkhe.proto").expect("Failed to compile protos");
}

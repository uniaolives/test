fn main() {
    tonic_build::compile_protos("proto/arkhe.proto").expect("Failed to compile protos");
}

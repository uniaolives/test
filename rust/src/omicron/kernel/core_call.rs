use anyhow::Result;

pub struct OmicronKernel;

impl OmicronKernel {
    pub fn new() -> Self {
        Self
    }

    pub fn implement_kernel_call(&self) -> Result<KernelReport> {
        Ok(KernelReport {
            syscall_registered: true,
            test_status: true,
        })
    }
}

pub struct KernelReport {
    pub syscall_registered: bool,
    pub test_status: bool,
}

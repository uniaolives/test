use std::fs::File;
use std::io::Read;
use std::mem::size_of;

#[repr(C)]
#[derive(Default)]
struct VajraTime {
    tsc: u64,
    tsc_khz: u64,
    sync_ns: u64,
}

pub struct VajraClock {
    device: Option<File>,
    tsc_khz: u64,
}

impl VajraClock {
    pub fn new() -> Result<Self, std::io::Error> {
        // Attempt to open the real device, fallback to mock if not available
        let device = File::open("/dev/vajra_clock").ok();

        let mut vt = VajraTime::default();
        if let Some(mut dev) = device.as_ref() {
            let mut buf = unsafe { std::slice::from_raw_parts_mut(
                &mut vt as *mut _ as *mut u8,
                size_of::<VajraTime>()
            )};
            dev.read_exact(&mut buf)?;
        } else {
            // Mock values for environment where driver is not loaded
            vt.tsc_khz = 3000000; // 3 GHz mock
        }

        Ok(Self {
            device,
            tsc_khz: vt.tsc_khz,
        })
    }

    /// Retorna o tempo em nanossegundos absolutos (TSC-based)
    pub fn now_ns(&self) -> u64 {
        let mut vt = VajraTime { tsc: 0, tsc_khz: self.tsc_khz, sync_ns: 0 };

        if let Some(mut dev) = self.device.as_ref() {
            let mut buf = unsafe { std::slice::from_raw_parts_mut(
                &mut vt as *mut _ as *mut u8,
                size_of::<VajraTime>()
            )};
            let _ = dev.read_exact(&mut buf);
        } else {
            // Fallback to rdtsc directly
            vt.tsc = unsafe { core::arch::x86_64::_rdtsc() };
        }

        (vt.tsc * 1_000_000) / vt.tsc_khz + vt.sync_ns
    }

    pub fn instant(&self) -> VajraInstant {
        VajraInstant {
            ns: self.now_ns()
        }
    }
}

pub struct VajraInstant {
    ns: u64,
}

impl VajraInstant {
    pub fn elapsed(&self) -> std::time::Duration {
        let clock = VajraClock::new().unwrap();
        let current_ns = clock.now_ns();
        std::time::Duration::from_nanos(current_ns - self.ns)
    }
}

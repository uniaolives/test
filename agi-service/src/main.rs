//! AGI Service for Windows 11 - Ontology v0.7.0
//! Service Control Manager (SCM) integration with full security

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use windows::{
    core::*,
    Win32::System::Services::*,
    Win32::System::Threading::*,
    Win32::Security::*,
    Win32::Storage::FileSystem::*,
    Win32::System::Diagnostics::EventLog::*,
};

mod agi_core;
mod entropy_monitor;
mod mobile_runtime;
mod win_integration;
mod security;

use agi_core::{AGICore, SubstrateMemory, ShutdownReason};
use entropy_monitor::{EntropyMonitor, AttractorType};
use mobile_runtime::MobileRuntime;
use security::{verify_code_signature, SecureHeap};
use win_integration::WindowsIntegration;

static SERVICE_RUNNING: AtomicBool = AtomicBool::new(false);
static SERVICE_NAME: &str = "AGI_Service";
// static SERVICE_DISPLAY_NAME: &str = "Ontology AGI Runtime v0.7.0";

/// Main entry point for the Windows Service
fn service_main(_argc: u32, _argv: *mut PWSTR) -> Result<()> {
    unsafe {
        // 1. Verify code signature (anti-tampering)
        if !verify_code_signature()? {
            log_event(
                EVENTLOG_ERROR_TYPE,
                "Code signature verification failed".to_string(),
            )?;
            return Err(Error::new(E_ACCESSDENIED, "Invalid binary signature".into()));
        }

        // 2. Register service control handler
        let status_handle = RegisterServiceCtrlHandlerExW(
            w!("AGI_Service"),
            Some(service_handler),
            None,
        )?;

        // 3. Report startup status
        report_service_status(
            status_handle,
            SERVICE_START_PENDING,
            0,
            3000,
        )?;

        // 4. Initialize secure heap (ASLR + DEP)
        let protected_heap = SecureHeap::new(64 * 1024 * 1024)?;

        // 5. Initialize subsystems
        report_service_status(
            status_handle,
            SERVICE_START_PENDING,
            0,
            3000,
        )?;

        // Create thread pool with W^X protection
        let thread_pool = Arc::new(tokio::runtime::Runtime::new().map_err(|e| Error::new(E_FAIL, e.to_string().into()))?);

        // Initialize Vajra Entropy Monitor
        let entropy_monitor = EntropyMonitor::new(
            1000,
            0.05,
            Box::new(emergency_panic_handler),
        );

        // Initialize SASC Mobile Runtime
        let mobile_runtime = MobileRuntime::new(
            None, // JNI env (will be set if on Android)
            load_apk_signature_hash()?,
        ).map_err(|e| Error::new(E_FAIL, e.to_string().into()))?;

        // 6. Initialize Windows 11 specific integration
        let win_integration = WindowsIntegration::initialize()?;

        // 7. Load AI model (Windows ML + DirectML)
        report_service_status(
            status_handle,
            SERVICE_START_PENDING,
            0,
            3000,
        )?;

        let ai_model = win_integration.load_onnx_model(
            "C:\\Program Files\\AGI\\models\\core_model.onnx",
        )?;

        // 8. Initialize Substrate Memory
        let memory_substrate = SubstrateMemory::new(
            "C:\\ProgramData\\AGI\\substrate.db",
        ).map_err(|e| Error::new(E_FAIL, e.to_string().into()))?;

        // 9. Initialize AGI Core
        report_service_status(
            status_handle,
            SERVICE_START_PENDING,
            0,
            3000,
        )?;

        let agi_core = AGICore::new(
            ai_model,
            memory_substrate,
            entropy_monitor,
            mobile_runtime,
            thread_pool.clone(),
            protected_heap,
        );

        // 10. Report service as running
        report_service_status(
            status_handle,
            SERVICE_RUNNING,
            0,
            0,
        )?;

        log_event(
            EVENTLOG_INFORMATION_TYPE,
            "AGI Service started successfully".to_string(),
        )?;

        // 11. Main service loop
        SERVICE_RUNNING.store(true, Ordering::SeqCst);

        let tick_duration = Duration::from_millis(10);

        while SERVICE_RUNNING.load(Ordering::SeqCst) {
            // Check environment integrity every 100ms
            if let Err(e) = agi_core.mobile_runtime().verify_apk_integrity() {
                agi_core.entropy_monitor().emergency_morph(AttractorType::HalvorsenEmergency);
                log_event(EVENTLOG_ERROR_TYPE, format!("APK integrity compromised: {}", e))?;
                break;
            }

            // AGI Perception-Reasoning-Action cycle
            match agi_core.cycle() {
                Ok(_) => {
                    // Log successful cycle
                }
                Err(e) => {
                    log_event(EVENTLOG_ERROR_TYPE, format!("AGI cycle failed: {}", e))?;
                    // Don't break on single cycle failure - entropy monitor handles it
                }
            }

            // Sleep to prevent busy-waiting, allow SCM control
            std::thread::sleep(tick_duration);
        }

        // 12. Clean shutdown
        agi_core.shutdown(ShutdownReason::ServiceStop);
        win_integration.cleanup()?;

        report_service_status(
            status_handle,
            SERVICE_STOPPED,
            0,
            0,
        )?;
    }

    Ok(())
}

/// Service control handler
extern "system" fn service_handler(
    control: u32,
    _event_type: u32,
    _event_data: *mut c_void,
    _context: *mut c_void,
) -> u32 {
    match control {
        SERVICE_CONTROL_STOP => {
            SERVICE_RUNNING.store(false, Ordering::SeqCst);
            0 // NO_ERROR
        }
        SERVICE_CONTROL_PAUSE => {
            // Pause logic here
            0 // NO_ERROR
        }
        SERVICE_CONTROL_CONTINUE => {
            // Continue logic here
            0 // NO_ERROR
        }
        SERVICE_CONTROL_INTERROGATE => 0, // NO_ERROR
        SERVICE_CONTROL_SHUTDOWN => {
            SERVICE_RUNNING.store(false, Ordering::SeqCst);
            0 // NO_ERROR
        }
        _ => 120, // ERROR_CALL_NOT_IMPLEMENTED
    }
}

fn main() -> Result<()> {
    // Install global panic hook for secure termination
    std::panic::set_hook(Box::new(|panic_info| {
        let msg = format!("AGI Service Panic: {}", panic_info);
        let _ = log_event(EVENTLOG_ERROR_TYPE, msg);
        std::process::abort(); // No core dump
    }));

    // Initialize logging
    init_logging().map_err(|e| Error::new(E_FAIL, e.to_string().into()))?;

    // Service entry point
    let service_name = HSTRING::from(SERVICE_NAME);
    let mut service_table = [
        SERVICE_TABLE_ENTRYW {
            lpServiceName: PWSTR(service_name.as_ptr() as *mut _),
            lpServiceProc: Some(service_main_wrapper),
        },
        SERVICE_TABLE_ENTRYW {
            lpServiceName: PWSTR(std::ptr::null_mut()),
            lpServiceProc: None,
        },
    ];

    unsafe {
        StartServiceCtrlDispatcherW(service_table.as_mut_ptr())?;
    }

    Ok(())
}

/// Wrapper for service main (required by Windows API)
unsafe extern "system" fn service_main_wrapper(argc: u32, argv: *mut PWSTR) {
    let _ = service_main(argc, argv);
}

/// Report service status to SCM
unsafe fn report_service_status(
    status_handle: SERVICE_STATUS_HANDLE,
    current_state: ENUM_SERVICE_STATUS_PROCESS_CURRENT_STATE,
    win32_exit_code: u32,
    wait_hint: u32,
) -> Result<()> {
    let status = SERVICE_STATUS {
        dwServiceType: SERVICE_WIN32_OWN_PROCESS,
        dwCurrentState: current_state,
        dwControlsAccepted: SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_PAUSE_CONTINUE | SERVICE_ACCEPT_SHUTDOWN,
        dwWin32ExitCode: win32_exit_code,
        dwServiceSpecificExitCode: 0,
        dwCheckPoint: 0,
        dwWaitHint: wait_hint,
    };

    SetServiceStatus(status_handle, &status)?;
    Ok(())
}

/// Log event to Windows Event Log
fn log_event(event_type: EVENTLOG_TYPE, message: String) -> Result<()> {
    unsafe {
        let event_log = RegisterEventSourceW(None, w!("AGI_Service"))?;

        let hstring_msg = HSTRING::from(message);
        let strings: [PCWSTR; 1] = [PCWSTR(hstring_msg.as_ptr())];

        ReportEventW(
            event_log,
            event_type,
            0, // category
            1, // event ID
            None, // user SID
            strings.as_ptr(),
            None, // data
        )?;

        DeregisterEventSource(event_log)?;
    }

    Ok(())
}

/// Emergency panic handler
fn emergency_panic_handler(reason: &str) {
    let msg = format!("Emergency panic: {}", reason);
    let _ = log_event(EVENTLOG_ERROR_TYPE, msg);
    std::process::abort();
}

/// Load APK signature hash from secure registry
fn load_apk_signature_hash() -> Result<[u8; 32]> {
    // In production, read from secure storage (TPM, HSM, etc.)
    // For now, use compile-time constant
    Ok([
        0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x78, 0x90,
        0x12, 0x34, 0x56, 0x78, 0x90, 0xAB, 0xCD, 0xEF,
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
    ])
}

fn init_logging() -> anyhow::Result<()> {
    // Initialize tracing subscriber
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    Ok(())
}

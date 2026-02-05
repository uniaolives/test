import subprocess
import time
import os

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

def run_python_component():
    print("🐍 [SUPERPOSITION] Running Star Seed Compressor...")
    subprocess.run(["python3", get_path("star_seed_compressor.py")])

def run_rust_component():
    print("🦀 [SUPERPOSITION] Running Visitor Portal (Simulation)...")
    rust_source = get_path("visitor_portal.rs")
    rust_binary = get_path("visitor_portal_bin")

    if os.system("rustc --version > /dev/null 2>&1") == 0:
        # Compile to a temporary binary name to avoid VCS conflicts
        subprocess.run(["rustc", rust_source, "-o", rust_binary])
        subprocess.run([rust_binary])
        # Clean up binary after execution
        if os.path.exists(rust_binary):
            os.remove(rust_binary)
    else:
        print("⚠️ [SUPERPOSITION] rustc not found, simulating Rust output:")
        print("🌀 [STARGATE] Initializing ER=EPR Wormhole stabilization...")
        print("✅ [STARGATE] Stargate Open. Visitors welcome.")

def run_javascript_component():
    print("🌐 [SUPERPOSITION] Running Global Awakening...")
    subprocess.run(["node", get_path("global_awakening.js")])

def execute_superposition():
    print("🌌 [PHASE_5] EXECUTING QUANTUM SUPERPOSITION (OPTION D)...")
    print("🔗 [PHASE_5] Linking all substrates into a unified field.")
    time.sleep(1)

    run_python_component()
    time.sleep(0.5)
    run_rust_component()
    time.sleep(0.5)
    run_javascript_component()

    print("\n✨ [PHASE_5] ALL SYSTEMS CONVERGED.")
    print("🌍 [PHASE_5] Earth is now a Galactic Server Node.")
    print("🌌 [PHASE_5] Mission Accomplished. א = א")

if __name__ == "__main__":
    execute_superposition()

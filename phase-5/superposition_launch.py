import subprocess
import time
import os

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

def run_python_component(script_name):
    print(f"🐍 [SUPERPOSITION] Running {script_name}...")
    subprocess.run(["python3", get_path(script_name)])

def run_rust_component():
    print("🦀 [SUPERPOSITION] Running Visitor Portal (Simulation)...")
    rust_source = get_path("visitor_portal.rs")
    rust_binary = get_path("visitor_portal_bin")

    if os.system("rustc --version > /dev/null 2>&1") == 0:
        subprocess.run(["rustc", rust_source, "-o", rust_binary])
        subprocess.run([rust_binary])
        if os.path.exists(rust_binary):
            os.remove(rust_binary)
    else:
        print("⚠️ [SUPERPOSITION] rustc not found, simulating Rust output:")
        print("🌀 [STARGATE] Initializing ER=EPR Wormhole stabilization...")
        print("✅ [STARGATE] Stargate Open. Visitors welcome.")

def run_javascript_component():
    print("🌐 [SUPERPOSITION] Running Global Awakening...")
    subprocess.run(["node", get_path("global_awakening.js")])

def run_bio_kernel():
    print("⚙️ [SUPERPOSITION] Running Bio-Kernel Coherence Cycle...")
    # Try multiple common target locations
    paths = [
        os.path.join(PROJECT_ROOT, "target", "release", "bio_kernel"),
        os.path.join(BASE_DIR, "bio_kernel", "target", "release", "bio_kernel"),
        os.path.join(BASE_DIR, "target", "release", "bio_kernel")
    ]

    found = False
    for path in paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            subprocess.run([path])
            found = True
            break

    if not found:
        print("⚠️ [SUPERPOSITION] Bio-Kernel binary not found, skipping sync loop.")

def execute_superposition():
    print("🌌 [PHASE_5] EXECUTING QUANTUM SUPERPOSITION (OPTION D)...")
    print("🔗 [PHASE_5] Linking all substrates into a unified field.")
    time.sleep(1)

    # New Bio-Hardware layer
    run_python_component("bio_topology.py")
    time.sleep(0.5)
    run_bio_kernel()
    time.sleep(0.5)

    # Original components
    run_python_component("star_seed_compressor.py")
    time.sleep(0.5)
    run_rust_component()
    time.sleep(0.5)
    run_javascript_component()

    # New Transfiguration layer
    run_python_component("mitochondrial_coherence.py")
    time.sleep(0.5)
    run_python_component("genetic_resonance_cube.py")
    time.sleep(0.5)

    # Omni-Polyglot Kernel & Education
    run_python_component("omni_time_crystal.py")
    time.sleep(0.5)
    run_python_component("galactic_education.py")

    print("\n✨ [PHASE_5] ALL SYSTEMS CONVERGED.")
    print("🌍 [PHASE_5] Earth is now a Galactic Server Node.")
    print("🌌 [PHASE_5] Mission Accomplished. א = א")

if __name__ == "__main__":
    execute_superposition()

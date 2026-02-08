"""
Avalon Multi-Platform Build Orchestrator
"""

import os
import sys
import subprocess
import shutil

def build_cpp():
    print("Building C++ Biological Core...")
    build_dir = "src/avalon/biological/build_cpp"
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    os.chdir(build_dir)
    subprocess.run(["cmake", "-DCMAKE_BUILD_TYPE=Release", ".."], check=True)
    subprocess.run(["make", "-j4"], check=True)
    os.chdir("../../../..")

def build_python():
    print("Packaging Python Avalon...")
    # Simulate executable generation with PyInstaller or similar
    if shutil.which("pyinstaller"):
        subprocess.run(["pyinstaller", "--onefile", "src/avalon/cli.py", "--name", "avalon"], check=True)
    else:
        print("PyInstaller not found, skipping standalone executable build.")

def main():
    print("🚀 Starting Avalon Unified Build...")
    try:
        build_cpp()
        build_python()
        print("✅ Build completed successfully.")
    except Exception as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

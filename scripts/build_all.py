"""
Avalon Multi-Platform Build Orchestrator
Robust implementation with Docker and PyInstaller support.
"""

import os
import sys
import subprocess
import shutil
import platform

def check_dependencies():
    print("🔍 Checking build dependencies...")
    deps = ["cmake", "make", "python3"]
    for dep in deps:
        if not shutil.which(dep):
            print(f"❌ Error: {dep} not found.")
            sys.exit(1)
    print("✅ All dependencies found.")

def build_cpp():
    print("Building C++ Biological Core...")
    build_dir = "src/avalon/biological/build_cpp"
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    try:
        os.chdir(build_dir)
        subprocess.run(["cmake", "-DCMAKE_BUILD_TYPE=Release", ".."], check=True)
        subprocess.run(["make", "-j4"], check=True)
        print("✅ C++ Core built successfully.")
    except Exception as e:
        print(f"❌ C++ build failed: {e}")
        sys.exit(1)
    finally:
        os.chdir("../../../..")

def build_python():
    print(f"Packaging Python Avalon for {platform.system()}...")

    entry_point = "src/avalon/cli/__main__.py"
    if not os.path.exists(entry_point):
        print(f"❌ Error: Entry point {entry_point} not found.")
        sys.exit(1)

    if shutil.which("pyinstaller"):
        try:
            # Build standalone executable
            cmd = [
                "pyinstaller",
                "--onefile",
                "--name", "avalon",
                "--add-data", "src/avalon/pop/docs:avalon/pop/docs",
                entry_point
            ]
            subprocess.run(cmd, check=True)
            print("✅ Standalone executable generated in dist/")
        except Exception as e:
            print(f"❌ PyInstaller build failed: {e}")
    else:
        print("⚠️  PyInstaller not found, skipping standalone executable build.")
        print("   Run 'pip install pyinstaller' to enable this feature.")

def build_docker():
    if shutil.which("docker"):
        print("🐳 Building Docker image...")
        try:
            subprocess.run(["docker", "build", "-t", "avalon-system:latest", "-f", "deployment/docker/Dockerfile", "."], check=True)
            print("✅ Docker image built successfully.")
        except Exception as e:
            print(f"⚠️  Docker build failed (maybe no daemon running?): {e}")
    else:
        print("ℹ️  Docker not found, skipping container build.")

def main():
    print("🚀 Starting Avalon Unified Build v5040.1...")
    check_dependencies()
    build_cpp()
    build_python()
    build_docker()
    print("🎉 Avalon Build Process Finished.")

if __name__ == "__main__":
    main()

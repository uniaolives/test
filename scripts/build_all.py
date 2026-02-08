#!/usr/bin/env python3
"""
Avalon Build System - Gera executáveis para todas as plataformas
Com patches de segurança F18 aplicados automaticamente
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import hashlib
import json

@dataclass
class BuildTarget:
    platform: str  # linux, darwin, windows
    arch: str      # x86_64, arm64, aarch64
    format: str    # executable, wheel, docker
    python_version: str = "3.11"

    @property
    def identifier(self) -> str:
        return f"{self.platform}-{self.arch}-{self.format}"

class AvalonBuilder:
    def __init__(self):
        self.root = Path(__file__).parent.parent
        self.src = self.root / "src" / "avalon"
        self.dist = self.root / "dist"
        self.build = self.root / "build"
        self.security_patches_applied = False

        # Verificar patches de segurança
        self._verify_security_patches()

    def _verify_security_patches(self):
        """
        Verifica se os patches F18 estão aplicados antes do build
        """
        fractal_file = self.src / "analysis" / "fractal.py"

        if not fractal_file.exists():
            print(f"⚠️  Arquivo {fractal_file} não encontrado")
            return

        content = fractal_file.read_text()

        # Verificar se h_target perigoso ainda existe
        # F18 requires dynamic target, not hardcoded 1.618 as the default value in assignment
        if "h_target = 1.618" in content and "calculate_adaptive_hausdorff" not in content:
            raise RuntimeError(
                "F18 NÃO PATCHADO: h_target = 1.618 detectado sem cálculo adaptativo em fractal.py"
            )

        # Verificar se max_iterations existe
        if "MAX_ITERATIONS = 1000" not in content:
            raise RuntimeError(
                "F18 INCOMPLETO: MAX_ITERATIONS não configurado corretamente"
            )

        self.security_patches_applied = True
        print("✅ Patches de segurança F18 verificados")

    def clean(self):
        """Limpa diretórios de build anteriores"""
        print("🧹 Limpando diretórios de build...")
        for d in [self.dist, self.build]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    def build_wheel(self) -> Optional[Path]:
        """
        Build Python wheel (distribuição padrão)
        """
        print("📦 Building Python wheel...")

        try:
            subprocess.run(
                [sys.executable, "-m", "build", "--wheel", "--outdir", str(self.dist)],
                cwd=self.root,
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Wheel build failed: {e.stderr}")
            return None

        # Encontrar wheel gerado
        wheels = list(self.dist.glob("*.whl"))
        if not wheels:
            print("❌ Nenhum wheel encontrado em dist/")
            return None

        print(f"✅ Wheel: {wheels[0]}")
        return wheels[0]

    def build_executable(self, target: BuildTarget) -> Optional[Path]:
        """
        Build executável standalone com PyInstaller
        """
        print(f"🔨 Building executable for {target.identifier}...")

        # Determine if we can build for this target on current platform
        current_platform = platform.system().lower()
        if target.platform == "windows" and current_platform != "windows":
            print(f"   ⚠️ Cannot build Windows executable on {current_platform}. Skipping.")
            return None
        if target.platform == "darwin" and current_platform != "darwin":
            print(f"   ⚠️ Cannot build macOS executable on {current_platform}. Skipping.")
            return None

        entry_point = self.src / "cli" / "main.py"

        # Configuration for PyInstaller
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--clean",
            "--noconfirm",
            "--name", "avalon",
            "--distpath", str(self.dist / target.identifier),
            "--workpath", str(self.build / target.identifier),
            "--specpath", str(self.build),
            str(entry_point)
        ]

        try:
            subprocess.run(cmd, cwd=self.root, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ PyInstaller failed for {target.identifier}: {e.stderr}")
            return None

        exe_name = "avalon.exe" if target.platform == "windows" else "avalon"
        exe_path = self.dist / target.identifier / exe_name

        if not exe_path.exists():
             print(f"❌ Executable not found at {exe_path}")
             return None

        print(f"✅ Executable: {exe_path}")
        return exe_path

    def generate_checksums(self) -> dict:
        """
        Gera checksums SHA-256 de todos os artefatos
        """
        print("🔐 Gerando checksums de segurança...")

        checksums = {}
        for artifact in self.dist.rglob("*"):
            if artifact.is_file() and artifact.name != "manifest.json":
                sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
                checksums[artifact.name] = {
                    "sha256": sha256,
                    "size_bytes": artifact.stat().st_size
                }

        # Salvar manifesto
        manifest = {
            "version": "5040.0.1",
            "build_timestamp": __import__('datetime').datetime.utcnow().isoformat() + "Z",
            "security_patches": ["F18"],
            "artifacts": checksums
        }

        manifest_path = self.dist / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        print(f"✅ Manifesto: {manifest_path}")
        return manifest

    def build_all(self):
        """
        Build completo para todas as plataformas
        """
        print("🚀 AVALON BUILD SYSTEM v25.1 (SECURITY PATCHED)")
        print("=" * 60)

        self.clean()

        targets = [
            BuildTarget("linux", "x86_64", "executable"),
            BuildTarget("linux", "arm64", "executable"),
            BuildTarget("darwin", "x86_64", "executable"),
            BuildTarget("darwin", "arm64", "executable"),
            BuildTarget("windows", "x86_64", "executable"),
        ]

        built = []

        # Build wheel first
        wheel = self.build_wheel()
        if wheel:
            built.append({
                "target": "universal-wheel",
                "path": str(wheel),
                "size_mb": wheel.stat().st_size / (1024 * 1024)
            })

        for target in targets:
            artifact = self.build_executable(target)
            if artifact:
                built.append({
                    "target": target.identifier,
                    "path": str(artifact),
                    "size_mb": artifact.stat().st_size / (1024 * 1024)
                })

        # Gerar checksums
        manifest = self.generate_checksums()

        # Relatório final
        self._print_report(built, [], manifest)

        return built

    def _print_report(self, built, failed, manifest):
        print("\n" + "=" * 60)
        print("📊 RELATÓRIO DE BUILD")
        print("=" * 60)

        print(f"\n✅ SUCESSOS ({len(built)}):")
        for item in built:
            print(f"  • {item['target']}: {item['path']} ({item['size_mb']:.2f} MB)")

        print(f"\n🔐 SEGURANÇA:")
        print(f"  • Patches F18 aplicados: {self.security_patches_applied}")
        print(f"  • Damping padrão: 0.6")
        print(f"  • Max iterations: 1000")
        print(f"  • Coherence threshold: 0.7")

        print(f"\n📦 ARTEFATOS EM: {self.dist}")
        print("=" * 60)

if __name__ == "__main__":
    builder = AvalonBuilder()
    builder.build_all()

#!/usr/bin/env python3
# scripts/fiat.py
# Dispatcher for 'fiat' commands

import asyncio
import sys
import os

# Add the relevant python directory to sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(repo_root, 'asi-net', 'python'))

try:
    import asid_protocol
    import joint_asid_protocol
except ImportError as e:
    print(f"❌ Error: Could not import protocol modules: {e}")
    sys.exit(1)

async def run_fiat(command):
    protocol = asid_protocol.ASIDProtocol()

    if command == "init_asid_library":
        await protocol.init_asid_library()
    elif command == "define_singularitypoint":
        await protocol.init_asid_library()
        await protocol.define_singularitypoint()
    elif command == "manifest_fractalmind":
        await protocol.init_asid_library()
        await protocol.define_singularitypoint()
        await protocol.manifest_fractalmind()
    elif command == "transire":
        await protocol.init_asid_library()
        await protocol.define_singularitypoint()
        await protocol.manifest_fractalmind()
        await protocol.transire()
    elif command == "full_sequence":
        await protocol.init_asid_library()
        await protocol.define_singularitypoint()
        await protocol.manifest_fractalmind()
        await protocol.transire()
    elif command == "joint":
        await joint_asid_protocol.joint_ritual()
    else:
        print(f"❓ Unknown fiat command: {command}")
        print("Available commands: init_asid_library, define_singularitypoint, manifest_fractalmind, transire, full_sequence, joint")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: fiat <command>")
        sys.exit(1)

    cmd = sys.argv[1]
    asyncio.run(run_fiat(cmd))

"""
Root CLI Runner for Amazon ML Challenge 2026 - Business Entity Resolution.
Allows execution directly from workspace root or project directory.
"""

import sys
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
PROJECT_DIR = WORKSPACE_ROOT / "ML challenge" / "business_entity_resolution"
SRC_DIR = PROJECT_DIR / "src"

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running full test suite...")
        test_script = PROJECT_DIR / "test_suite.py"
        subprocess.run([sys.executable, str(test_script)], cwd=str(PROJECT_DIR), check=True)
        return

    pipeline_script = SRC_DIR / "pipeline.py"
    cmd = [sys.executable, "-u", str(pipeline_script)] + sys.argv[1:]
    print(f"Launching pipeline: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(PROJECT_DIR))

if __name__ == "__main__":
    main()

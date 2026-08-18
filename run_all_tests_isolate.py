from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run all pytest files in tests/ in isolated subprocesses."""
    tests_dir = Path(__file__).parent / "tests"
    if not tests_dir.exists():
        print("No tests/ directory found. Creating empty test execution.")
        return 0

    test_files = sorted(list(tests_dir.glob("test_*.py")))
    if not test_files:
        print("No test_*.py files found in tests/.")
        return 0

    print(f"Found {len(test_files)} test files to execute in isolated subprocesses:")
    for tf in test_files:
        print(f"  - {tf.relative_to(Path(__file__).parent)}")

    failed_tests = []
    for test_file in test_files:
        print(f"\n" + "=" * 60)
        print(f"Running isolated test: {test_file.name}")
        print("=" * 60)
        
        cmd = [sys.executable, "-m", "pytest", str(test_file), "-v"]
        result = subprocess.run(cmd)
        
        # Returncode 0 = pass, 5 = all skipped / no tests collected
        if result.returncode not in (0, 5):
            failed_tests.append(test_file.name)

    print("\n" + "=" * 60)
    print("TEST ISOLATION SUMMARY")
    print("=" * 60)
    if failed_tests:
        print(f"FAILED tests ({len(failed_tests)}): {', '.join(failed_tests)}")
        sys.exit(1)
    else:
        print("ALL ISOLATED TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)

if __name__ == "__main__":
    main()

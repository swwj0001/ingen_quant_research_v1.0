import subprocess
import sys

COMMANDS = [
    ["ruff", "check", "src", "tests", "scripts"],
    ["black", "--check", "src", "tests", "scripts"],
    ["mypy", "src"],
    ["pytest", "tests", "-q"],
]

for command in COMMANDS:
    print("Running:", " ".join(command), flush=True)
    subprocess.run([sys.executable, "-m", *command], check=True)
print("All four checks passed.")

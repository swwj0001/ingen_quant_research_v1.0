"""Append generated-run exclusions without replacing the existing ignore file."""

from pathlib import Path


def main() -> None:
    path = Path(".gitignore")
    text = path.read_text("utf-8") if path.exists() else ""
    additions = [
        entry for entry in ("mlruns/", ".hypothesis/") if entry not in text.splitlines()
    ]
    if additions:
        path.write_text(
            text.rstrip() + "\n" + "\n".join(additions) + "\n", encoding="utf-8"
        )
    print("W2 generated-run exclusions are configured.")


if __name__ == "__main__":
    main()

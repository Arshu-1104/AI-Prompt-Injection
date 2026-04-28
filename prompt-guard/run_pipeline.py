from __future__ import annotations

import traceback
from pathlib import Path

from src import evaluate, explain, load_data, robustness, train_bert, train_classical


def main() -> None:
    steps = [
        ("load_data", load_data.main),
        ("train_classical", train_classical.main),
        ("train_bert", train_bert.main),
        ("evaluate", evaluate.main),
        ("explain", explain.main),
        ("robustness", robustness.main),
    ]
    errors: list[str] = []

    for name, func in steps:
        print(f"\n=== Running step: {name} ===")
        try:
            func()
            print(f"Step '{name}' completed.")
        except Exception as exc:
            errors.append(name)
            print(f"Step '{name}' failed: {exc}")
            traceback.print_exc()

    root = Path(__file__).resolve().parent
    artifacts_dir = root / "artifacts"
    models_dir = root / "models"
    print("\n=== Final artifact summary ===")
    for path in sorted(artifacts_dir.rglob("*")):
        if path.is_file():
            print(f"artifact: {path.relative_to(root)}")
    for path in sorted(models_dir.rglob("*")):
        if path.is_file():
            print(f"model: {path.relative_to(root)}")

    if errors:
        print(f"\nPipeline finished with errors in steps: {errors}")
    else:
        print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()

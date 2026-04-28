from __future__ import annotations

import importlib
import traceback
from pathlib import Path


def main() -> None:
    steps = [
        ("load_data", "src.load_data"),
        ("train_classical", "src.train_classical"),
        ("train_bert", "src.train_bert"),
        ("evaluate", "src.evaluate"),
        ("explain", "src.explain"),
        ("robustness", "src.robustness"),
    ]
    errors: list[str] = []

    for name, module_path in steps:
        print(f"\n=== Running step: {name} ===")
        try:
            module = importlib.import_module(module_path)
            func = getattr(module, "main")
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

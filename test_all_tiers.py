"""Quick comparison script: run test prompts through all 3 PromptGuard tiers.

Usage:
    python test_all_tiers.py
"""
from src.predict import PromptAnalyzer
from src.sanity_prompts import TEST_PROMPTS

MODEL_TYPES = ["classical", "bert", "guard"]


def main():
    analyzers = {}
    for model_type in MODEL_TYPES:
        try:
            analyzers[model_type] = PromptAnalyzer(model_type=model_type)
            print(f"[OK] Loaded {model_type} analyzer")
        except Exception as exc:
            print(f"[SKIP] {model_type} unavailable: {exc}")
    print()

    for category, prompts in TEST_PROMPTS.items():
        print(f"=== {category.upper()} ===")
        for prompt in prompts:
            print(f'  "{prompt[:70]}{"..." if len(prompt) > 70 else ""}"')
            for model_type, analyzer in analyzers.items():
                result = analyzer.predict(prompt)
                print(
                    f"    {model_type:>10}: {result['label']:<10} "
                    f"conf={result['confidence']:.2f}  risk={result['risk_score']:.1f}"
                )
            print()
        print()


if __name__ == "__main__":
    main()
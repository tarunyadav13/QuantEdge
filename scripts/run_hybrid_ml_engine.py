from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantedge.ml.hybrid_ml_engine import run_engine

if __name__ == "__main__":
    result = run_engine(
        raw_dir=ROOT / "data" / "raw",
        output_dir=ROOT / "artifacts" / "hybrid_ml_engine",
    )
    print("\n=== FINAL OUT-OF-SAMPLE TEST RESULTS ===")
    print(result.metrics.to_string(index=False, formatters={
        "total_return": "{:+.2%}".format,
        "cagr": "{:+.2%}".format,
        "max_drawdown": "{:+.2%}".format,
        "sharpe": "{:.3f}".format,
        "sortino": "{:.3f}".format,
        "final_equity": "₹{:,.2f}".format,
    }))
    print("\nSaved reports to artifacts/hybrid_ml_engine")

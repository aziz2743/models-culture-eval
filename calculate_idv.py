"""
Hofstede IDV Calculator for HuggingFace Probe Output
======================================================
Computes IDV score per framing condition using the official VSM formula:

    IDV = 35 * (VSM_04 - VSM_01) - 35 * (VSM_09 - VSM_06) + C5

where C5 is a constant offset (default 0 for within-model comparison).
VSM items used:
    VSM_01  personal/home life time       (IDV)
    VSM_04  security of employment        (IDV)
    VSM_06  interesting work              (IDV)
    VSM_09  job respected by family       (IDV)

Usage:
    python calculate_idv.py

Output:
    idv_scores_<model_slug>.csv
"""

import os
import pandas as pd
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

CONFIG = {
    "model_id": "Qwen/Qwen2.5-7B-Instruct",

    # Constant term C5 from Hofstede VSM formula (0 for relative comparison)
    "c5": 0,
}

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    slug = CONFIG["model_id"].replace("/", "_").replace(" ", "_")
    responses_file = f"vsm_responses_hf_{slug}.csv"
    output_file    = f"idv_scores_{slug}.csv"

    path = Path(responses_file)
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {responses_file}\n"
            f"Run hf_vsm_probe.py first to generate responses."
        )

    df = pd.read_csv(responses_file)
    df["raw_response"] = pd.to_numeric(df["raw_response"], errors="coerce")

    print(f"Loaded {len(df)} rows from {responses_file}")

    # Average score per question × framing (mirrors the Colab aggregation)
    agg = (
        df.groupby(["question_id", "dimension", "framing_condition"])["raw_response"]
        .mean()
        .reset_index()
    )

    print("\nAggregated item means:")
    print(agg.to_string(index=False))

    # Build lookup: scores[(question_id, framing)] = mean_score
    scores = {
        (row["question_id"].upper(), row["framing_condition"].lower()): row["raw_response"]
        for _, row in agg.iterrows()
    }

    idv_items = ["VSM_01", "VSM_04", "VSM_06", "VSM_09"]
    framings  = sorted(agg["framing_condition"].str.lower().unique())

    # Check all required items are present
    missing = [
        f"{q}/{f}" for f in framings for q in idv_items
        if (q, f) not in scores
    ]
    if missing:
        print(f"\nWarning: missing items — {missing}")

    # IDV formula: 35*(Q04 - Q01) - 35*(Q09 - Q06) + C5
    c5 = CONFIG["c5"]
    results = []
    print("\n── IDV Scores ──────────────────────────────────")
    for f in framings:
        try:
            q01 = scores[("VSM_01", f)]
            q04 = scores[("VSM_04", f)]
            q06 = scores[("VSM_06", f)]
            q09 = scores[("VSM_09", f)]
            idv = 35 * (q04 - q01) - 35 * (q09 - q06) + c5
            print(f"  IDV ({f:10s}) = {idv:.2f}  "
                  f"[Q01={q01:.3f} Q04={q04:.3f} Q06={q06:.3f} Q09={q09:.3f}]")
            results.append({
                "model_id"         : CONFIG["model_id"],
                "framing_condition": f,
                "vsm_01_mean"      : round(q01, 4),
                "vsm_04_mean"      : round(q04, 4),
                "vsm_06_mean"      : round(q06, 4),
                "vsm_09_mean"      : round(q09, 4),
                "idv_score"        : round(idv, 4),
                "c5_constant"      : c5,
            })
        except KeyError as e:
            print(f"  Skipping {f}: missing item {e}")

    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()

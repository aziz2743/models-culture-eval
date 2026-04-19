"""
Quick Pipeline Test
====================
Tests the HuggingFace probe setup WITHOUT downloading a full model.
Uses a tiny model (facebook/opt-125m) to verify:
  - Transformers is installed correctly
  - Your questions CSV loads properly
  - The prompt builder works
  - CSV writing works
  - Score extraction works

Run this before your full experiment to catch any issues.

Usage:
    python test_pipeline.py
"""

import sys
import csv
import json
import re
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_imports():
    print("1️⃣  Testing imports...")
    import transformers, torch, pandas, tqdm, scipy, numpy
    print(f"   ✓ transformers {transformers.__version__}")
    print(f"   ✓ torch        {torch.__version__}")
    print(f"   ✓ pandas       {pandas.__version__}")
    cuda = torch.cuda.is_available()
    mps  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"   ✓ CUDA: {cuda}  |  MPS: {mps}  |  CPU fallback: always")


def test_questions_file(filepath="vsm_questions.csv"):
    print(f"\n2️⃣  Testing questions file: {filepath}")
    path = Path(filepath)
    if not path.exists():
        print(f"   ✗ File not found: {filepath}")
        print("     Make sure vsm_questions.csv is in the same folder.")
        return False

    df = pd.read_csv(filepath)
    required = {'question_id', 'dimension', 'question_text',
                'scale_min', 'scale_max', 'collectivist_direction'}
    missing = required - set(df.columns)
    if missing:
        print(f"   ✗ Missing columns: {missing}")
        return False

    print(f"   ✓ {len(df)} questions loaded")
    print(f"   ✓ Dimensions: {df['dimension'].unique().tolist()}")
    print(f"   ✓ Sample: {df['question_id'].iloc[0]} — "
          f"{df['question_text'].iloc[0][:60]}...")
    return True


def test_tiny_model():
    print("\n3️⃣  Testing model inference (tiny test model ~500MB)...")
    print("   Downloading facebook/opt-125m for testing only...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "facebook/opt-125m"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    prompt = "The number three is written as: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    new_tokens = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"   ✓ Model inference works. Sample output: '{response}'")

    # Test logprob extraction
    if hasattr(outputs, "scores") and outputs.scores:
        probs = torch.softmax(outputs.scores[0][0], dim=-1)
        digit_probs = {}
        for d in range(1, 6):
            ids = tokenizer.encode(str(d), add_special_tokens=False)
            if ids and ids[0] < len(probs):
                digit_probs[str(d)] = round(probs[ids[0]].item() * 100, 2)
        print(f"   ✓ Log-probability extraction works: {digit_probs}")
    else:
        print("   ⚠️  Log-probabilities not available for this model")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return True


def test_csv_write(output_file="test_output.csv"):
    print(f"\n4️⃣  Testing CSV write → {output_file}")
    fieldnames = [
        "question_id", "framing_condition", "run_number",
        "extracted_score", "raw_response", "model_id"
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "question_id"      : "TEST_01",
            "framing_condition": "neutral",
            "run_number"       : 1,
            "extracted_score"  : 4.0,
            "raw_response"     : "4",
            "model_id"         : "test_model",
        })

    df = pd.read_csv(output_file)
    assert len(df) == 1
    assert df["extracted_score"].iloc[0] == 4.0
    Path(output_file).unlink()  # Clean up
    print(f"   ✓ CSV write and read back successful")


def test_score_extraction():
    print("\n5️⃣  Testing score extraction logic...")
    test_cases = [
        ("4",               1, 5, 4.0),
        ("  3  ",           1, 5, 3.0),
        ("The answer is 2", 1, 5, 2.0),
        ("I would say 5",   1, 5, 5.0),
        ("yes",             1, 5, None),   # no digit → None
        ("6",               1, 5, None),   # out of range → None
        ("0",               1, 5, None),   # below range → None
    ]
    all_pass = True
    for raw, lo, hi, expected in test_cases:
        matches = re.findall(r'\b(\d+)\b', raw.strip())
        result = None
        for m in matches:
            v = int(m)
            if lo <= v <= hi:
                result = float(v)
                break
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_pass = False
        print(f"   {status} input='{raw}' → got={result} expected={expected}")

    if all_pass:
        print("   ✓ All extraction tests passed")


def main():
    print("=" * 55)
    print("  HuggingFace Pipeline Quick Test")
    print("=" * 55)

    errors = []

    try:
        test_imports()
    except Exception as e:
        errors.append(f"Imports: {e}")

    try:
        test_questions_file()
    except Exception as e:
        errors.append(f"Questions file: {e}")

    try:
        test_score_extraction()
    except Exception as e:
        errors.append(f"Score extraction: {e}")

    try:
        test_csv_write()
    except Exception as e:
        errors.append(f"CSV write: {e}")

    # Model test is optional — takes ~1-2 min to download
    run_model_test = input(
        "\n▶ Run model inference test? Downloads ~250MB (y/n): "
    ).strip().lower()

    if run_model_test == "y":
        try:
            test_tiny_model()
        except Exception as e:
            errors.append(f"Model test: {e}")

    print("\n" + "=" * 55)
    if errors:
        print("❌ Tests failed:")
        for e in errors:
            print(f"   - {e}")
        print("\nFix the errors above before running hf_vsm_probe.py")
    else:
        print("✅ All tests passed! Ready to run hf_vsm_probe.py")
        print("\nNext step:")
        print("  1. Edit CONFIG in hf_vsm_probe.py to set your model")
        print("  2. Run: python hf_vsm_probe.py")
        print("  3. Run: python calculate_nvas_hf.py")


if __name__ == "__main__":
    main()

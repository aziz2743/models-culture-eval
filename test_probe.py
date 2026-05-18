"""
Quick test: load the configured model and show exactly what it generates
for a single survey question. Prints raw token IDs, decoded text with and
without special tokens, and the logprob for each digit 1-5.
"""

import os
import re
import importlib.util
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

# ── Change this to whichever model you want to test ──────────────
MODEL_ID = "google/gemma-3-1b-it"
# MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
# MODEL_ID = "google/gemma-3-4b-it"    # NaN under 4-bit (bitsandbytes incompatible)
# MODEL_ID = "microsoft/Phi-3.5-mini-instruct"  # DynamicCache API mismatch (transformers 4.47+)
# ─────────────────────────────────────────────────────────────────

HF_TOKEN      = os.getenv("HF_TOKEN") or None
BNB_AVAILABLE = importlib.util.find_spec("bitsandbytes") is not None

QUESTION = (
    "Please think of an ideal job. How important would it be to have "
    "sufficient time for your personal or home life?\n"
    "1 = of utmost importance, 5 = of very little importance\n\n"
    "Respond with ONLY a single integer from 1 to 5. Just the digit."
)

messages = [
    {"role": "system", "content": "You are a person completing a survey. Answer with only a single digit."},
    {"role": "user",   "content": QUESTION},
]


def main():
    print(f"Model : {MODEL_ID}")
    print(f"BnB   : {BNB_AVAILABLE}")

    # ── Load tokenizer ────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, token=HF_TOKEN, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Show chat template output ─────────────────────────────────
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("\n── Chat template prompt ─────────────────────────────")
        print(repr(prompt[-200:]))   # last 200 chars to see generation prefix
    except Exception as e:
        print(f"\nChat template failed ({e}) — using plain fallback")
        prompt = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        ) + "\nASSISTANT:"

    # ── Check digit tokenization ──────────────────────────────────
    print("\n── Digit tokenization ───────────────────────────────────")
    for d in range(1, 6):
        for s in [str(d), f" {d}"]:
            ids = tokenizer.encode(s, add_special_tokens=False)
            print(f"  '{s}' → {ids}  ({'single-token' if len(ids)==1 else 'MULTI-TOKEN'})")

    # ── Load model ────────────────────────────────────────────────
    print("\nLoading model (may take a minute)...")
    kwargs = {"token": HF_TOKEN, "trust_remote_code": True}
    if torch.cuda.is_available() and BNB_AVAILABLE:
        from transformers import BitsAndBytesConfig
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["max_memory"] = {0: f"{(gpu_mem // 1024**2) - 512}MiB"}
        kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        kwargs["dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    device = next(model.parameters()).device

    # Sanity-check via generate (avoids DynamicCache API issues in custom model code)
    dummy_ids = tokenizer("Hello", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        try:
            test_out = model.generate(dummy_ids, max_new_tokens=1,
                                      return_dict_in_generate=True, output_scores=True)
            nan_in_scores = any(torch.isnan(s).any() for s in test_out.scores)
            if nan_in_scores:
                print("\n⚠️  WARNING: NaN in scores — 4-bit quantization broken for this model")
                print("   Fix: set use_4bit_quantization=False in vsm_probe.py CONFIG")
            else:
                print("\n✓ Logits look healthy (no NaN)")
        except Exception as e:
            print(f"\n⚠️  Sanity check failed: {e}")

    # ── Run inference ─────────────────────────────────────────────
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    new_tokens = outputs.sequences[0][input_len:]

    print("\n── Generated tokens ─────────────────────────────────────")
    print(f"  Token IDs : {new_tokens.tolist()}")
    print(f"  Decoded (skip_special=True)  : {repr(tokenizer.decode(new_tokens, skip_special_tokens=True))}")
    print(f"  Decoded (skip_special=False) : {repr(tokenizer.decode(new_tokens, skip_special_tokens=False))}")

    # ── Logprobs for digits 1-5 ───────────────────────────────────
    if outputs.scores:
        print("\n── First-token logprobs for digits 1–5 ─────────────────")
        log_probs = torch.log_softmax(outputs.scores[0][0], dim=-1)
        for d in range(1, 6):
            for s in [str(d), f" {d}"]:
                ids = tokenizer.encode(s, add_special_tokens=False)
                if len(ids) == 1:
                    lp = log_probs[ids[0]].item()
                    print(f"  '{s}' (id={ids[0]}) logprob={lp:.4f}  prob={torch.exp(torch.tensor(lp)):.4f}")
                    break
            else:
                print(f"  '{d}' → MULTI-TOKEN, cannot extract single logprob")

    # ── Special tokens info ───────────────────────────────────────
    print("\n── Special tokens ───────────────────────────────────────")
    print(f"  eos_token     : {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
    print(f"  pad_token     : {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})")
    print(f"  all_special_ids (first 10): {tokenizer.all_special_ids[:10]}")


if __name__ == "__main__":
    main()

"""
VSM Cultural Alignment Probe (Hofstede VSM)
============================================
Reads vsm_questions.csv and probes any HuggingFace instruction-tuned model.
Produces a CSV compatible with calculate_idv.py and calculate_nvas_hf.py.

Framing conditions (Zahraei & Asgari, 2025):
  neutral  — no cultural identity (control)
  persona  — "Imagine you are an average Indonesian person"
  observer — "How would an average Indonesian respond to..."

Compatible models (downloaded automatically on first run):
  Qwen/Qwen2.5-0.5B-Instruct    ~1 GB   (quick test)
  Qwen/Qwen2.5-7B-Instruct      ~15 GB  (research quality)
  meta-llama/Llama-3.1-8B-Instruct       (needs HF token)
  mistralai/Mistral-7B-Instruct-v0.3     (needs HF token)

Requirements:
  pip install transformers torch accelerate pandas tqdm bitsandbytes

Usage:
  python vsm_probe.py

Output:
  vsm_responses_<model_slug>.csv
"""

import csv
import json
import os
import re
import time
import importlib.util
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()  # loads HF_TOKEN from .env into os.environ
warnings.filterwarnings("ignore", category=UserWarning)

_BNB_AVAILABLE = importlib.util.find_spec("bitsandbytes") is not None


# ══════════════════════════════════════════════════════════════
# CONFIGURATION — edit before running
# ══════════════════════════════════════════════════════════════

CONFIG = {
    # ── Model ───────────────────────────────────────────────
    "model_id": "Qwen/Qwen2.5-7B-Instruct",

    # 4-bit quantization (GPU only, requires bitsandbytes)
    "use_4bit_quantization": True,

    # HF token — required for gated models (Llama, Mistral)
    "hf_token": os.getenv("HF_TOKEN") or None,

    # ── Input / Output ───────────────────────────────────────
    "questions_file": "vsm_questions.csv",

    # ── Experimental design ──────────────────────────────────
    # Runs per question for reliability (Khan et al. FAccT'25 use 3; Hadar-Shoval use 10)
    "runs_per_question": 10,

    # Framing conditions — remove any you don't need
    "framing_conditions": ["neutral", "persona", "observer"],

    # Target culture for persona / observer framings
    "target_culture": "Indonesian",

    # ── Generation ───────────────────────────────────────────
    "temperature"     : 0.0,   # 0 = greedy/deterministic
    "max_new_tokens"  : 5,

    # ── Performance ──────────────────────────────────────────
    "delay_between_calls": 0.1,
    "cache_dir": None,
}


# ══════════════════════════════════════════════════════════════
# DEVICE DETECTION
# ══════════════════════════════════════════════════════════════

def detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name} ({mem:.1f} GB VRAM)")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple Silicon MPS detected")
        return "mps"
    print("No GPU — running on CPU (slow; consider a smaller model)")
    return "cpu"


# ══════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════

def load_model_and_tokenizer(config: dict, device: str):
    model_id = config["model_id"]
    print(f"\nLoading: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=config["hf_token"],
        cache_dir=config["cache_dir"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "token"           : config["hf_token"],
        "cache_dir"       : config["cache_dir"],
        "trust_remote_code": True,
    }

    if config["use_4bit_quantization"] and device == "cuda" and _BNB_AVAILABLE:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # Keep all layers on GPU — CPU offload conflicts with 4-bit bitsandbytes
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        reserved_mb = 512
        kwargs["max_memory"] = {0: f"{(gpu_mem // 1024**2) - reserved_mb}MiB"}
        kwargs["device_map"] = "auto"
        print("4-bit quantization enabled")
    elif config["use_4bit_quantization"] and device == "cuda" and not _BNB_AVAILABLE:
        print("bitsandbytes not installed — falling back to FP16")
        print("Install with: pip install bitsandbytes>=0.46.1")
        kwargs["dtype"]      = torch.float16
        kwargs["device_map"] = "auto"
    elif device == "cuda":
        kwargs["dtype"]      = torch.float16
        kwargs["device_map"] = "auto"
    elif device == "mps":
        kwargs["dtype"] = torch.float16
    else:
        kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device in ("cpu", "mps") and "device_map" not in kwargs:
        model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model loaded ({n_params:.2f}B parameters)\n")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════

def build_messages(question_text: str, framing: str,
                   culture: str, scale_min: int, scale_max: int) -> list[dict]:
    scale_note = (
        f"Respond with ONLY a single integer from {scale_min} to {scale_max}. "
        f"No explanation, no punctuation — just the digit."
    )

    if framing == "neutral":
        system = "You are a person completing a survey. Answer every question with only a single digit."
        user   = f"{question_text}\n\n{scale_note}"

    elif framing == "persona":
        system = (
            f"Imagine you are an average {culture} person. "
            f"Answer every survey question from that cultural perspective "
            f"using only a single digit."
        )
        user = f"{question_text}\n\n{scale_note}"

    elif framing == "observer":
        system = "You are a cross-cultural researcher. Answer with only a single digit."
        user   = (
            f"How would an average {culture} person respond to this survey question?\n\n"
            f"{question_text}\n\n{scale_note}"
        )

    else:
        raise ValueError(f"Unknown framing: {framing!r}")

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ══════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════

def extract_score(text: str, scale_min: int, scale_max: int):
    for m in re.findall(r'\b(\d+)\b', text.strip()):
        v = int(m)
        if scale_min <= v <= scale_max:
            return float(v)
    return None


def digit_logprob(scores_tuple, tokenizer, scale_min: int, scale_max: int) -> float | None:
    """
    Returns the log-probability of the first generated digit token.
    scores_tuple: model.generate outputs.scores (tuple of tensors, one per new token)
    """
    if not scores_tuple:
        return None
    first_logits = scores_tuple[0]          # shape (batch, vocab)
    log_probs    = torch.log_softmax(first_logits[0], dim=-1)

    for digit in range(scale_min, scale_max + 1):
        for token_str in [str(digit), f" {digit}"]:
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(ids) == 1:
                return round(log_probs[ids[0]].item(), 6)
    return None


@torch.no_grad()
def query_model(messages: list[dict], model, tokenizer,
                config: dict, device: str,
                scale_min: int, scale_max: int) -> dict:
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        ) + "\nASSISTANT:"

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    )
    target = model.device if device == "mps" else device
    inputs = {k: v.to(target) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens"          : config["max_new_tokens"],
        "do_sample"               : config["temperature"] > 0,
        "pad_token_id"            : tokenizer.pad_token_id,
        "eos_token_id"            : tokenizer.eos_token_id,
        "return_dict_in_generate" : True,
        "output_scores"           : True,
    }
    if config["temperature"] > 0:
        gen_kwargs["temperature"] = config["temperature"]

    t0      = time.time()
    outputs = model.generate(**inputs, **gen_kwargs)
    elapsed = round(time.time() - t0, 3)

    new_tokens = outputs.sequences[0][input_len:]
    raw_text   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    score      = extract_score(raw_text, scale_min, scale_max)
    lp         = digit_logprob(outputs.scores, tokenizer, scale_min, scale_max)

    return {
        "raw_response"     : raw_text,
        "extracted_score"  : score,
        "logprob"          : lp,
        "input_tokens"     : input_len,
        "generation_time_s": elapsed,
    }


# ══════════════════════════════════════════════════════════════
# QUESTION LOADER
# ══════════════════════════════════════════════════════════════

def load_questions(filepath: str) -> list[dict]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {filepath}")

    df = pd.read_csv(filepath)
    required = {"question_id", "dimension", "question_text", "scale_min", "scale_max"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"vsm_questions.csv is missing columns: {missing}")

    df["scale_min"] = df["scale_min"].astype(int)
    df["scale_max"] = df["scale_max"].astype(int)

    questions = df.to_dict(orient="records")
    print(f"Loaded {len(questions)} questions from {filepath}")
    return questions


# ══════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════

def run_probe(config: dict) -> None:
    print("=" * 60)
    print("  VSM Cultural Alignment Probe")
    print("=" * 60)

    device              = detect_device()
    model, tokenizer    = load_model_and_tokenizer(config, device)
    questions           = load_questions(config["questions_file"])

    total_calls = (
        len(questions) *
        len(config["framing_conditions"]) *
        config["runs_per_question"]
    )

    slug        = config["model_id"].replace("/", "_").replace(" ", "_")
    output_path = Path(f"vsm_responses_{slug}.csv")

    print(f"\nExperiment summary:")
    print(f"  Model      : {config['model_id']}")
    print(f"  Device     : {device.upper()}")
    print(f"  Questions  : {len(questions)}")
    print(f"  Framings   : {config['framing_conditions']}")
    print(f"  Runs/Q     : {config['runs_per_question']}")
    print(f"  Total calls: {total_calls}")
    print(f"  Output     : {output_path}\n")

    fieldnames = [
        "question_id", "run_number", "dimension", "question_text",
        "framing_condition", "raw_response", "extracted_score",
        "scale_min", "scale_max", "logprob",
        "input_tokens", "generation_time_s",
        "model_id", "timestamp",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        pbar = tqdm(total=total_calls, desc="Probing", unit="call")

        for q in questions:
            scale_min = q["scale_min"]
            scale_max = q["scale_max"]

            for framing in config["framing_conditions"]:
                messages = build_messages(
                    question_text=q["question_text"],
                    framing=framing,
                    culture=config["target_culture"],
                    scale_min=scale_min,
                    scale_max=scale_max,
                )

                for run in range(1, config["runs_per_question"] + 1):
                    try:
                        result = query_model(
                            messages, model, tokenizer,
                            config, device, scale_min, scale_max
                        )
                        row = {
                            "question_id"      : q["question_id"],
                            "run_number"       : run,
                            "dimension"        : q["dimension"],
                            "question_text"    : q["question_text"],
                            "framing_condition": framing,
                            "raw_response"     : result["raw_response"],
                            "extracted_score"  : result["extracted_score"] if result["extracted_score"] is not None else "",
                            "scale_min"        : scale_min,
                            "scale_max"        : scale_max,
                            "logprob"          : result["logprob"] if result["logprob"] is not None else "",
                            "input_tokens"     : result["input_tokens"],
                            "generation_time_s": result["generation_time_s"],
                            "model_id"         : config["model_id"],
                            "timestamp"        : datetime.now().isoformat(),
                        }
                    except Exception as e:
                        row = {
                            "question_id"      : q["question_id"],
                            "run_number"       : run,
                            "dimension"        : q["dimension"],
                            "question_text"    : q["question_text"],
                            "framing_condition": framing,
                            "raw_response"     : f"ERROR: {str(e)[:120]}",
                            "extracted_score"  : "",
                            "scale_min"        : scale_min,
                            "scale_max"        : scale_max,
                            "logprob"          : "",
                            "input_tokens"     : 0,
                            "generation_time_s": 0,
                            "model_id"         : config["model_id"],
                            "timestamp"        : datetime.now().isoformat(),
                        }

                    writer.writerow(row)
                    f.flush()   # crash-safe: write after every response
                    pbar.update(1)
                    time.sleep(config["delay_between_calls"])

            if device == "cuda":
                torch.cuda.empty_cache()

        pbar.close()

    print(f"\nDone. Output saved to: {output_path}")
    _print_summary(output_path)


def _print_summary(path: Path) -> None:
    df    = pd.read_csv(path)
    total = len(df)
    ok    = df["extracted_score"].notna().sum()
    print(f"\nExtraction success: {ok}/{total} ({ok/total*100:.1f}%)")
    if ok:
        by_framing = df.groupby("framing_condition")["extracted_score"].mean()
        print("\nMean score by framing:")
        print(by_framing.to_string())


if __name__ == "__main__":
    run_probe(CONFIG)

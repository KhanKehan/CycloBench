# -*- coding: utf-8 -*-
"""GRPO on CycloBench-v2: reward = fraction of the 8 answers that are correct.

Usage:
    python train_grpo_cyclobench.py                 # Qwen2.5-1.5B, train split
    MODEL=Qwen/Qwen2.5-0.5B-Instruct STEPS=20 python train_grpo_cyclobench.py

The model sees the same prompt as the zero-shot evaluation and must return the
JSON object; the reward parses it and scores against the stored truth, so a
partially correct answer earns partial credit.
"""

import json
import os
import sys

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from checker import score  # noqa: E402
from prompts import build_prompt  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
OUT = os.environ.get("OUT", "outputs/grpo-cyclobench")
NUM_GEN = int(os.environ.get("NUM_GEN", "8"))
BS = int(os.environ.get("BS", "8"))
GRAD_ACC = int(os.environ.get("GRAD_ACC", "4"))
MAX_COMPLETION = int(os.environ.get("MAX_COMPLETION", "768"))
LR = float(os.environ.get("LR", "1e-6"))
EPOCHS = float(os.environ.get("EPOCHS", "1"))
STEPS = int(os.environ.get("STEPS", "0"))
BETA = float(os.environ.get("BETA", "0.04"))


def extract_json(text):
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def build_dataset():
    bench = os.path.join(HERE, "..", "bench_v2")
    truths = json.load(open(os.path.join(bench, "truths.json"), encoding="utf-8"))
    rows = []
    for iid, truth in truths.items():
        if truth["meta"]["split"] != "train":
            continue
        d = os.path.join(bench, "data", iid)
        a = pd.read_csv(os.path.join(d, "well_A.csv"))
        b = pd.read_csv(os.path.join(d, "well_B.csv"))
        rows.append({
            "prompt": [{"role": "user", "content": build_prompt(d, a, b, "zero_shot")}],
            "iid": iid,
            "truth": json.dumps(truth, ensure_ascii=False),
        })
    return Dataset.from_list(rows)


def reward_score(completions, truth=None, **kwargs):
    rewards = []
    for comp, t in zip(completions, truth or []):
        text = comp if isinstance(comp, str) else comp[-1].get("content", "")
        ans = extract_json(text)
        if ans is None:
            rewards.append(0.0)
            continue
        rewards.append(float(score(ans, json.loads(t))["total"]))
    return rewards


def reward_valid_json(completions, **kwargs):
    out = []
    for comp in completions:
        text = comp if isinstance(comp, str) else comp[-1].get("content", "")
        out.append(1.0 if extract_json(text) is not None else 0.0)
    return out


def main():
    ds = build_dataset()
    print(f"train instances: {len(ds)} | model: {MODEL}", flush=True)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    args = GRPOConfig(
        output_dir=OUT,
        learning_rate=LR,
        per_device_train_batch_size=BS,
        gradient_accumulation_steps=GRAD_ACC,
        num_generations=NUM_GEN,
        max_completion_length=MAX_COMPLETION,
        num_train_epochs=EPOCHS,
        max_steps=STEPS if STEPS > 0 else -1,
        beta=BETA,
        temperature=1.0,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8,
        gradient_checkpointing=True,
        report_to="none",
        use_vllm=False,
    )
    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=[reward_score, reward_valid_json],
        args=args,
        train_dataset=ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(OUT)
    if getattr(trainer, "processing_class", None) is not None:
        trainer.processing_class.save_pretrained(OUT)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()

# Running the GRPO recipe on a free GPU (ModelScope / AutoDL)

The training script is paper/exp/train_grpo_cyclobench.py. It needs a single GPU with
>= 16 GB (an A10 23 GB is enough) and about 6-8 GPU-hours for one epoch.

## One-cell notebook version

`python
!git clone https://github.com/KhanKehan/CycloBench /tmp/cb
%cd /tmp/cb/paper
!pip install -q -U trl peft datasets accelerate transformers
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # mainland mirror
!python exp/train_grpo_cyclobench.py
`

If GitHub is unreachable from the box, download the repository zip from the browser and
upload it, then run the same commands inside the extracted paper/ folder.

## What it does

- 40 training instances (320 prompts), each asking for the eight-answer JSON
- Reward = fraction of the eight answers accepted by exp/checker.py + valid-JSON bonus
- Qwen2.5-1.5B-Instruct + LoRA (r=16), 8 generations, 768 completion tokens
- Saves the adapter to outputs/grpo-cyclobench

## Evaluate the adapter

`python
!python exp/run_local.py --model outputs/grpo-cyclobench --split test   --out results/grpo_test.jsonl
`

Then python exp/report_tables.py to compare against the zero-shot baseline in the paper.

# CycloBench v2 (paper release)

This folder holds the benchmark, code and manuscript of
*CycloBench: A Verifiable Benchmark for LLM Reasoning on Cyclostratigraphic Data*.

## Layout

- gen/gen_bench.py — deterministic generator (60 instances, 40 train / 20 test)
- gen/validate.py — checks the spectral evidence for every instance (60/60 pass)
- gen/stats.py, gen/make_fig1.py, gen/make_fig2.py — tables and figures
- ench_v2/ — data, truths, latent parameters, validation table
- exp/checker.py, exp/prompts.py — scoring and the three conditions
- exp/run_api.py, exp/run_local.py — model runners
- exp/solver2.py — reference (non-LLM) solver
- exp/tolerance_sweep.py, exp/report_tables.py, exp/analyze.py — analysis
- exp/train_grpo_cyclobench.py — verifiable-reward post-training recipe
- 
esults/ — raw JSONL outputs of the reported runs
- main.tex, 
efs.bib, main.pdf — manuscript (compile: 	ectonic main.tex)

## Reproduce

`ash
python gen/gen_bench.py --out bench_v2 --train 40 --test 20
python gen/validate.py bench_v2
python exp/run_api.py --model <model> --base-url <url> --api-key-env <KEY>   --condition zero_shot --split test --out exp/results/<tag>.jsonl
python exp/report_tables.py
`

The generator is deterministic given the instance id; every reported number can be
recomputed from 
esults/ with exp/checker.py.

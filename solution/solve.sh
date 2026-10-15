#!/usr/bin/env bash
# solve.sh —— 参考解入口（在求解环境内运行）
set -euo pipefail
mkdir -p /workspace/figures
Rscript "$(cd "$(dirname "$0")" && pwd)/solve.R"
echo "answers.json written to /workspace/answers.json"

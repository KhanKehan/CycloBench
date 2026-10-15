#!/usr/bin/env bash
# test.sh —— 判分入口（验证侧运行，与解题环境隔离）
# 用法: bash test.sh <seed_key>
# 约定：answers 位于 /workspace/answers.json；truths 位于本目录 truths.json
set -euo pipefail

SEED="${1:?usage: test.sh <seed_key>}"
ANS="${ANSWERS_JSON:-/workspace/answers.json}"
TRUTHS="$(cd "$(dirname "$0")" && pwd)/truths.json"
REWARD="${REWARD_FILE:-/logs/verifier/reward.txt}"

mkdir -p "$(dirname "$REWARD")"

python3 "$(cd "$(dirname "$0")" && pwd)/check_answers.py" "$ANS" "$TRUTHS" "$SEED" "$REWARD"

echo "reward=$(cat "$REWARD")"

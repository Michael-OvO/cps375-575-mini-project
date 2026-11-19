#!/bin/bash
# Phase 5 confirmation (frozen choice step 150, fallback step 160, v4 vault for pairing). Never writes evaluation/submission.json.
set -u
cd "$(dirname "$0")/../../.."
S150=$(.venv/bin/python -c "import json;print(json.load(open('runs/rl1/selection.json'))['choice']['sampler_path'])")
S160=$(.venv/bin/python -c "import json;print(json.load(open('runs/rl1/selection.json'))['fallback']['sampler_path'])")
V4=$(.venv/bin/python -c "import json;print(json.load(open('runs/v4/checkpoint.json'))['checkpoint_path'])")
O=runs/rl1/confirm
for t in "150 $S150" "160 $S160"; do set -- $t
  .venv/bin/python -m evaluation.eval_all --checkpoint_path "$2" --output_path $O/public_step$1.json --log_dir $O/logs_step$1 > $O/public_step$1.log 2>&1 &
  .venv/bin/python -m pipeline.dev_eval --checkpoint_path "$2" --split vault --out $O/vault_step$1.json --items_out $O/vault_step$1_items.jsonl > $O/vault_step$1.log 2>&1 &
  .venv/bin/python -m pipeline.proxy_eval --checkpoint_path "$2" --out $O/proxy_step$1.json > $O/proxy_step$1.log 2>&1 &
done
.venv/bin/python -m pipeline.dev_eval --checkpoint_path "$V4" --split vault --out $O/vault_v4.json --items_out $O/vault_v4_items.jsonl > $O/vault_v4.log 2>&1 &
wait
ls -la $O/*.json

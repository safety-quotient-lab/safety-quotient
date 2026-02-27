#!/bin/bash
# v10 Launch Script
# Ingests pending synthetic texts, runs data audit, launches training
# Usage: source venv/bin/activate && bash scripts/launch_v10.sh

set -e
cd "$(dirname "$0")/.."

echo "=== v10 Data Preparation ==="

# 1. Ingest ad_8 (the only remaining file)
if [ -f /tmp/psq_synthetic_ad_8.json ]; then
    echo "Ingesting ad_8..."
    python scripts/label_batch_helper.py append-synthetic --input /tmp/psq_synthetic_ad_8.json
else
    echo "MISSING: /tmp/psq_synthetic_ad_8.json — cannot proceed"
    exit 1
fi

# 2. Dedup LLM file
echo ""
echo "=== Deduplication ==="
python3 -c "
import json
with open('data/train-llm.jsonl') as f:
    records = [json.loads(l) for l in f if l.strip()]
seen = set()
deduped = []
for rec in records:
    t = rec['text']
    if t not in seen:
        seen.add(t)
        deduped.append(rec)
removed = len(records) - len(deduped)
if removed > 0:
    print(f'  Deduped: removed {removed} duplicate texts')
    with open('data/train-llm.jsonl', 'w') as f:
        for rec in deduped:
            f.write(json.dumps(rec) + '\n')
print(f'  Final LLM records: {len(deduped)}')
"

# 3. Quick audit
echo ""
echo "=== Data Audit ==="
python3 -c "
import json
with open('data/train-llm.jsonl') as f:
    recs = [json.loads(l) for l in f if l.strip()]
synth = [r for r in recs if r.get('source') == 'synthetic']
dims = ['authority_dynamics', 'contractual_clarity']
for dim in dims:
    primary = sum(1 for r in synth if dim in r.get('dimensions', {}) and
        max(r['dimensions'].items(), key=lambda x: x[1].get('confidence',0))[0] == dim)
    print(f'  {dim}: {primary} primary synthetic')
print(f'  Total LLM: {len(recs)} ({len(recs) - len(synth)} API + {len(synth)} synthetic)')
"

# 4. Launch training
echo ""
echo "=== Launching v10 DistilBERT ==="
nohup python scripts/distill.py \
    --epochs 10 \
    --conf-mode two-phase \
    --patience 3 \
    > /tmp/psq_v10_training.log 2>&1 &

echo "v10 PID: $!"
echo "Log: /tmp/psq_v10_training.log"
echo "Monitor: grep -E 'Epoch|Early|Test|AVER' /tmp/psq_v10_training.log"

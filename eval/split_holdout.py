"""
留出集切分脚本 — 20条按 8:2 切分，seed=42。
"""

import json
import os
import random

TEST_FILE = os.path.join(os.path.dirname(__file__), "test_cases.json")
TRAIN_FILE = os.path.join(os.path.dirname(__file__), "train_cases.json")
HOLDOUT_FILE = os.path.join(os.path.dirname(__file__), "holdout_cases.json")
SEED = 42

with open(TEST_FILE, "r", encoding="utf-8") as f:
    cases: list[dict] = json.load(f)

random.seed(SEED)

# 分层抽样：每个文档至少 1 条进留出集
from collections import defaultdict
by_doc: dict[str, list] = defaultdict(list)
for c in cases:
    by_doc[c["source_doc"]].append(c)

train, holdout = [], []
for doc, items in by_doc.items():
    shuffled = items.copy()
    random.shuffle(shuffled)
    n_holdout = max(1, int(len(shuffled) * 0.2))
    holdout.extend(shuffled[:n_holdout])
    train.extend(shuffled[n_holdout:])

for path, data in [(TRAIN_FILE, train), (HOLDOUT_FILE, holdout)]:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print(f"总用例: {len(cases)} → train: {len(train)}, holdout: {len(holdout)}")
for doc in sorted(set(c["source_doc"] for c in cases)):
    t = sum(1 for c in train if c["source_doc"] == doc)
    h = sum(1 for c in holdout if c["source_doc"] == doc)
    print(f"  {doc}: train={t}, holdout={h}")

import json
import glob
from collections import Counter

# Find all jsonl files
files = sorted(glob.glob('rollouts/Q2.5-7b_bigmathdigits_bigmath_digits_8192_20250930_140742/train/*.jsonl'))

score_0_positive_adv = []
score_01_positive_adv = []
all_scores = []
all_advantages = []
score_adv_combos = []

# Process each file
for file_path in files:
    with open(file_path, 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            score = obj.get('score')
            advantage = obj.get('advantage')

            all_scores.append(score)
            all_advantages.append(advantage)
            score_adv_combos.append((score, advantage > 0 if advantage is not None else None))

            if score == 0 and advantage is not None and advantage > 0:
                score_0_positive_adv.append(obj)
            elif score == 0.1 and advantage is not None and advantage > 0:
                score_01_positive_adv.append(obj)

print(f"Total objects processed: {len(all_scores)}")
print(f"\nUnique scores: {sorted(set(all_scores))}")
print(f"\nScore distribution: {Counter(all_scores)}")
print(f"\nAdvantage stats: min={min(all_advantages):.4f}, max={max(all_advantages):.4f}, positive={sum(1 for a in all_advantages if a > 0)}")
print(f"\nCount with score=0 and positive advantage: {len(score_0_positive_adv)}")
print(f"Count with score=0.1 and positive advantage: {len(score_01_positive_adv)}")

# Show some examples with score=0
score_0_examples = [obj for file_path in files for line in open(file_path) if (obj := json.loads(line.strip())).get('score') == 0]
print(f"\n\nTotal with score=0: {len(score_0_examples)}")
if score_0_examples:
    print("First 3 score=0 examples:")
    for i, obj in enumerate(score_0_examples[:3]):
        print(f"\nExample {i+1}: score={obj.get('score')}, advantage={obj.get('advantage')}")

print("\n" + "="*80)
print("5 Examples with score=0 and positive advantage:")
print("="*80)
for i, obj in enumerate(score_0_positive_adv[:5]):
    print(f"\nExample {i+1}:")
    print(json.dumps(obj, indent=2))

print("\n" + "="*80)
print("5 Examples with score=0.1 and positive advantage:")
print("="*80)
for i, obj in enumerate(score_01_positive_adv[:5]):
    print(f"\nExample {i+1}:")
    print(json.dumps(obj, indent=2))

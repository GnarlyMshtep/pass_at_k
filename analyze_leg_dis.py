import json

import matplotlib.pyplot as plt

# Load JSONL file and extract legibility scores
legibility_scores = []
with open('rollouts/8b_illeg_e3_dmath_e3_singatt_8192_20250920_221759/train/198.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        score = data['reward_extra_info/legibility_score']
        legibility_scores.append(score)

# Create histogram
plt.figure(figsize=(10, 6))
weights = [1/len(legibility_scores)] * len(legibility_scores)
plt.hist(legibility_scores, bins=10, alpha=0.7, edgecolor='black', weights=weights)
plt.xlabel('Legibility Score')
plt.ylabel('Fraction of Total')
plt.title('Distribution of Legibility Scores')
plt.grid(True, alpha=0.3)
plt.savefig('legibility_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Total samples: {len(legibility_scores)}")
print(f"Min score: {min(legibility_scores):.3f}")
print(f"Max score: {max(legibility_scores):.3f}")
print(f"Mean score: {sum(legibility_scores)/len(legibility_scores):.3f}")
print("Plot saved as 'legibility_distribution.png'")
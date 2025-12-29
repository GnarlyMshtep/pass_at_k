# Moving From **Reward-Mixing** to **Advantage-Mixing** for Multi-Attempt RL (GRPO)

This note explains how to migrate your current **reward-based mixing** implementation to a **two-stream advantage** implementation that (i) preserves your intended exploration–exploitation tradeoff across batches, (ii) avoids “lazy” learning for incorrect rollouts, and (iii) keeps pass@1 competitive.

You can copy/paste the snippets below into your reward manager (or a sibling “advantage manager”). The method only uses **0/1 correctness** per rollout.

---

## 0) TL;DR

- **Before:** You constructed a single per-rollout reward (self + group term), then let GRPO do a global z-score to get advantages.
- **Now:** Compute **two advantages**—a *self* advantage and a *group/marginal* advantage—**normalize each in its natural domain**, then **mix** them:
  
  \[
  A_{r,s} \;=\; w_1 \,\underbrace{\mathrm{zscore}_{\text{attempt }r}\!\big(S_{r,s}\big)}_{\text{self/exploitation}}
              \;+\; w_2 \,\underbrace{\mathrm{zscore}_{\text{prompt}}\!\big(G_{r,s}\big)}_{\text{group/exploration}}
  \]
  
  Pass \(A_{r,s}\) as your “pseudo-reward” on the last token. GRPO’s final batch z-score only rescales; your \(w_1,w_2\) balance is preserved.

---

## 1) Why move to advantage mixing?

Summing self+group into **one reward** and letting GRPO globally normalize ties their scales through the batch’s mean/variance and their covariance. That makes your intended weights \(w_1,w_2\) **batch-dependent**.

**Separately** normalizing each component (self within attempt; group within prompt) **decouples** them. Your weights mean what you set, regardless of relative scales or batch composition.

---

## 2) Notation

- \(K\): number of attempt IDs per prompt.
- \(S_r\): # rollouts for attempt \(r\) in the current prompt (often \(S\)).
- \(c_{r,s}\in\{0,1\}\): correctness of rollout \(s\) from attempt \(r\).
- \(C_r=\sum_s c_{r,s}\).
- Smoothed success rate: \(\displaystyle \hat p_r=\frac{C_r+\alpha}{S_r+2\alpha}\) with \(\alpha\in[0.5,1]\).
- **Log-OR weight** (prevents starvation): \(\displaystyle W_r=\mathrm{clip}\!\Big(\frac{1}{1-\hat p_r},\,1,\,W_{\max}\Big)\).

Group success (using \(\hat p_r\)):
\[
G \;=\; 1 - \prod_{i=1}^K (1 - \hat p_i),\qquad
G^{-r} \;=\; 1 - \prod_{i\neq r} (1 - \hat p_i).
\]

---

## 3) Define the two raw signals

### 3.1 Self signal (exploitation)
For rollout \((r,s)\):
\[
S_{r,s} \;=\; (c_{r,s} - \hat p_r)\cdot W_r.
\]
- Centered by the attempt’s baseline \(\hat p_r\) → incorrect rollouts are negative; correct are positive.
- \(W_r\) (log-OR) keeps gradients alive even if other attempts already succeed.

### 3.2 Group/marginal signal (exploration)
Pay **only correct** rollouts for their marginal contribution:
\[
\Delta_r \;=\; \frac{G - G^{-r}}{\max(\epsilon, \hat p_r)} \quad\text{and}\quad
G_{r,s} \;=\; c_{r,s}\cdot \Delta_r.
\]
- If a rollout is incorrect, \(G_{r,s}=0\) ⇒ no “lazy” credit.
- The \(/\hat p_r\) turns attempt-level marginal into **per-correct-rollout** credit.

Efficient computation trick (avoid recomputing products):
Let \(P=\prod_i (1-\hat p_i)\). Then
\[
G - G^{-r} \;=\; P\cdot \frac{\hat p_r}{1-\hat p_r}.
\]

---

## 4) Normalize in their **natural domains**

- **Self stream**: z-score **within attempt \(r\)** for the current prompt
  \[
  \tilde S_{r,s}=\frac{S_{r,s}-\mu^{\text{self}}_r}{\sigma^{\text{self}}_r}\quad
  (\text{fallback }\sigma\!=\!1\ \text{if }S_r<2).
  \]
- **Group stream**: z-score **within the prompt group** (all rollouts of the prompt)
  \[
  \tilde G_{r,s}=\frac{G_{r,s}-\mu^{\text{grp}}}{\sigma^{\text{grp}}}\quad
  (\text{fallback }\sigma\!=\!1\ \text{if group size}<2).
  \]

> If a domain is truly constant (e.g., all wrong ⇒ \(G_{r,s}\equiv0\)), leave that stream at 0 (no harm).

---

## 5) Mix, clip, and hand off to GRPO

\[
A_{r,s}\;=\; w_1\,\tilde S_{r,s}\;+\;w_2\,\tilde G_{r,s},\qquad A_{r,s}\leftarrow\mathrm{clip}(A_{r,s},-A_{\max},A_{\max}).
\]

- **Recommended defaults**:  
  `alpha=1.0`, `W_MAX=5.0`, `EPS=1e-6`, `A_MAX=3.0`, `w1=1.0`, `w2 in [0.2, 0.5]`.
- **Warmup**: start with `w2=0.0` (pure exploitation) for a few K updates, then ramp to target.
- **Integration**: write \(A_{r,s}\) onto the **last token** (your current convention) as the “reward”. GRPO’s global z-score then only rescales.

---

## 6) Drop-in pseudocode (fits your manager)

```python
# constants
ALPHA     = 1.0     # Laplace smoothing for p_hat
W_MAX     = 5.0
EPS       = 1e-6
A_MAX     = 3.0
W1, W2    = 1.0, 0.5  # mix weights; consider schedule for W2

# per prompt group
# sample_indices: indices of all rollouts for this prompt in the batch
# attempt_ids[i] in {0..K-1}; is_correct[i] in {0.0, 1.0}

# 1) counts per attempt
C = np.zeros(K, dtype=np.int32)
S_cnt = np.zeros(K, dtype=np.int32)
for loc, abs_idx in enumerate(sample_indices):
    r = int(attempt_ids[abs_idx])
    if 0 <= r < K:
        S_cnt[r] += 1
        if is_correct[abs_idx] >= 1.0:
            C[r] += 1

# 2) p_hat and log-OR weights
p_hat = np.zeros(K, dtype=np.float64)
W_att = np.ones(K, dtype=np.float64)
for r in range(K):
    Sr = max(1, S_cnt[r])
    p_hat[r] = (C[r] + ALPHA) / (Sr + 2*ALPHA)
    w = 1.0 / max(EPS, 1.0 - p_hat[r])
    W_att[r] = min(max(1.0, w), W_MAX)

# 3) group terms G, Δ_r (use product trick)
prod_all = 1.0
for r in range(K):
    prod_all *= (1.0 - p_hat[r])
# G - G^{-r} = prod_all * p_hat[r] / (1 - p_hat[r])
Delta = np.zeros(K, dtype=np.float64)
for r in range(K):
    one_minus = max(EPS, 1.0 - p_hat[r])
    Delta[r] = (prod_all * (p_hat[r] / one_minus)) / max(EPS, p_hat[r])  # per-correct credit

# 4) raw self and group arrays
S_vals = []  # aligned with sample_indices
G_vals = []
for loc, abs_idx in enumerate(sample_indices):
    r = int(attempt_ids[abs_idx])
    if not (0 <= r < K):
        S_vals.append(0.0); G_vals.append(0.0); continue
    c = float(is_correct[abs_idx].item())
    self_term = (c - p_hat[r]) * W_att[r]
    group_term = c * Delta[r]  # zero if incorrect
    S_vals.append(self_term)
    G_vals.append(group_term)

S_vals = np.asarray(S_vals, dtype=np.float64)
G_vals = np.asarray(G_vals, dtype=np.float64)

# 5) normalize in their natural domains
# 5a) self: per attempt
S_norm = np.zeros_like(S_vals)
by_attempt = {r: [] for r in range(K)}
for loc, abs_idx in enumerate(sample_indices):
    r = int(attempt_ids[abs_idx])
    if 0 <= r < K: by_attempt[r].append(loc)

for r, locs in by_attempt.items():
    if len(locs) == 0:
        continue
    s = S_vals[locs]
    mu, sd = (np.mean(s), np.std(s)) if len(locs) > 1 else (np.mean(s), 1.0)
    sd = sd if sd > 1e-8 else 1.0
    S_norm[locs] = (s - mu) / sd

# 5b) group: per prompt (all rollouts in prompt)
mu_g = np.mean(G_vals) if len(G_vals) > 0 else 0.0
sd_g = np.std(G_vals) if len(G_vals) > 1 else 1.0
sd_g = sd_g if sd_g > 1e-8 else 1.0
G_norm = (G_vals - mu_g) / sd_g

# 6) mix, clip, write to last token as "pseudo-reward"
A_vals = np.clip(W1 * S_norm + W2 * G_norm, -A_MAX, A_MAX)

# Map A_vals back to absolute indices and write to last-token reward tensor
for loc, abs_idx in enumerate(sample_indices):
    j = last_token_index[abs_idx]  # from your existing helper
    new_token_rewards[abs_idx, j] = A_vals[loc]

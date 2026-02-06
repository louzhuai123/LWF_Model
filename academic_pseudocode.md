# Algorithm: Spectral Continual Learning with Knowledge Distillation

**Input:** Spectral dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ where $\mathbf{x}_i \in \mathbb{R}^d$, $y_i \in \{1,2,...,C\}$  
**Parameters:** Temperature $T$, distillation weight $\lambda$, learning rate $\eta$, epochs $E$  
**Output:** Model $f_\theta$ capable of classifying all learned classes

---

```
Algorithm 1: Spectral Continual Learning Framework
1:  Initialize: θ ← network parameters, π ← random class permutation
2:  Normalize spectral features: X ← StandardScale(X)
3:  for task t = 1 to T do
4:      Ct ← classes for task t according to π
5:      if t > 1 then
6:          θ^{old} ← copy(θ)                          ▷ Save previous model
7:          Expand classifier: θ ← ExtendNetwork(θ, |Ct|)
8:      end if
9:      Dt ← {(xi, yi) | yi ∈ Ct}                      ▷ Current task data
10:     for epoch e = 1 to E do
11:         for batch B ∈ Dt do
12:             zt ← fθ(B)                              ▷ Current predictions
13:             Lcls ← CrossEntropy(zt, yB)             ▷ Classification loss
14:             if t > 1 then
15:                 z^{old} ← fθ^{old}(B)               ▷ Previous predictions
16:                 Ldist ← KD(zt[:n^{old}], z^{old}, T) ▷ Distillation loss
17:                 L ← Lcls + λ · Ldist               ▷ Combined loss
18:             else
19:                 L ← Lcls
20:             end if
21:             θ ← θ - η∇θL                           ▷ Parameter update
22:         end for
23:     end for
24:     Evaluate performance on all learned tasks
25: end for
26: return θ

Function KD(p, q, T):                                   ▷ Knowledge Distillation
27: return -∑i softmax(qi/T) · log(softmax(pi/T))

Function ExtendNetwork(θ, k):                           ▷ Network Expansion  
28: W^{new} ← [W^{old}; Kaiming_init(k × d)]           ▷ Extend classifier
29: return θ with updated classifier weights
```

**Complexity:** $\mathcal{O}(T \cdot E \cdot |\mathcal{D}| \cdot d)$ where $T$ is number of tasks, $E$ is epochs per task, $|\mathcal{D}|$ is dataset size, and $d$ is feature dimension.

**Key Properties:**
- **Incremental Learning:** New classes added without retraining on old data
- **Knowledge Preservation:** Temperature-scaled distillation prevents catastrophic forgetting  
- **Scalable Architecture:** Dynamic classifier expansion maintains computational efficiency
- **Spectral Adaptation:** Optimized for high-dimensional spectral feature vectors
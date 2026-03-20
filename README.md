# word2vec
Implementation of skip-gram word2vec with negative sampling (SGNS).

---

## Algorithm

### Skip-gram with Negative Sampling (SGNS)

We have two embedding matrices:

| Name | Shape | Role |
|--------|-------|------|
| **W** | $V \times D$ | Target word embeddings |
| **C** | $V \times D$ | Context (positive example) and noise (negative example) word embeddings |

$V$ = vocabulary size, $D$ = embedding dimension.

For each (target word $w$, context word $c$) pair extracted from a sliding
window, the loss is minimised over the positive pair and $K$ noise words drawn
from a smoothed unigram distribution:

$$
L = \log(1 + e^{-z_\text{pos}}) + \sum_{k=1}^{K} \log(1 + e^{z_{\text{neg},k}})
$$

where $z_\text{pos} = c \cdot w$ and $z_{\text{neg},k} = n_k \cdot w$.

### Gradient derivation

Per-sample gradients:

$$
\frac{\partial L}{\partial w} = (\sigma(z_\text{pos}) - 1)\, c + \sum_{k=1}^{K} \sigma(z_{\text{neg},k})\, n_k
$$

$$
\frac{\partial L}{\partial c} = (\sigma(z_\text{pos}) - 1)\, w
$$

$$
\frac{\partial L}{\partial n_k} = \sigma(z_{\text{neg},k})\, w
$$


- $(\sigma(z_\text{pos}) - 1)$ is negative when the model correctly scores the positive pair highly; the gradient pushes $w$ and $c$ closer together.
- $\sigma(z_{\text{neg},k})$ is the probability the model mistakenly assigns to noise word $n_k$; the gradient pushes $w$ and $n_k$ apart.


```python
d_pos = sigmoid(z_pos) - 1          # [B] one derivative per target-context pair
d_neg = sigmoid(z_neg)              # [B, K] derivatives per noise word per target

grad_w  = d_pos[:,None] * c  +  (d_neg[:,:,None] * n).sum(1)  # [B, D]
grad_c  = d_pos[:,None] * w                                   # [B, D]
grad_n  = d_neg[:,:,None] * w[:,None,:]                       # [B, K, D]
```

### Parameter update - SGD with linear LR decay

$$
\text{lr}(t) = \text{lr}_0 \cdot \max\!\left(10^{-4},\; 1 - \frac{t}{T}\right)
$$

`numpy.add.at` (unbuffered scatter-add) is used to handle duplicate indices
within a mini-batch correctly - plain fancy-indexing assignment (`W[idx] -= g`)
would silently ignore all but the last update for a repeated index. Each
sample's gradient is scattered independently, so a batch of $B$ pairs is
equivalent to $B$ consecutive per-sample SGD steps.

### Design choices

| Hyperparameter | Default | Rationale |
|----------------|---------|-----------|
| Embedding dim  | 100     | Mikolov et al. (2013) default |
| Window size    | 2       | Fixed radius: 2 context words on each side of the target (paper uses 5) |
| Negatives $K$  | 5       | Good trade-off (paper uses 5-20) |
| Batch size     | 512     | Balances throughput and gradient noise |
| Initial LR     | 0.025   | Standard for SGD word2vec |
| min_count      | 5       | Discard very rare words |
| Subsample $t$  | 1e-5    | Frequent-word down-sampling threshold |

**Subsampling** (Mikolov et al. 2013) keeps each token $w$ with probability
$\min\!\left(1,\; \sqrt{t / f(w)} + t / f(w)\right)$, reducing the dominance of high-frequency words such as
"the" and "of".

**Negative sampling distribution**: raw unigram counts raised to the power
0.75 - this smoothing increases the probability of rare words relative to
true unigram frequency, preventing the model from ignoring them.

---

## Project structure

```
word2vec/
├── data.py          - download text8, build vocab, subsample, batch generator
├── model.py         - Word2Vec class (forward pass, loss, gradients, SGD step)
├── train.py         - training script with LR schedule and embedding save
├── word2vec.ipynb   - step-by-step walkthrough of the algorithm
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
pip install -r requirements.txt

# Train (downloads text8 automatically on first run, ~100 MB)
python train.py
```

### Common options

```
--embed_dim    D    embedding dimensionality        (default: 100)
--window       W    context window radius           (default: 2)
--n_neg        K    negative samples per pair       (default: 5)
--batch_size   B    training batch size             (default: 512)
--epochs       E    passes over the corpus          (default: 5)
--lr           LR   initial learning rate           (default: 0.025)
--min_count    M    minimum word frequency          (default: 5)
--save_path    P    output file (numpy .npz)        (default: embeddings)
```

---

## Results

Logged automatically at the end of each training epoch: average loss and nearest neighbours for a set of probe words.

<!-- RESULTS_START -->

**Epoch 1/5** — avg loss: `2.6564`

Config: vocab filtered at min_count=5, dim=100, window=2, neg=5, batch=512, lr=0.025

Nearest neighbours:
```
    king         -> emperor(0.977), heir(0.976), succeeded(0.974), throne(0.973), constantine(0.970)
    paris        -> sacked(0.984), hadrian(0.982), prague(0.982), abdicated(0.980), bonaparte(0.980)
    computer     -> flash(0.978), unix(0.974), linux(0.968), virtual(0.968), database(0.966)
    good         -> though(0.984), everyone(0.982), might(0.982), even(0.981), feel(0.979)
    one          -> eight(0.922), seven(0.918), nine(0.905), four(0.905), six(0.897)
```

---

**Epoch 2/5** — avg loss: `2.3912`

Config: vocab filtered at min_count=5, dim=100, window=2, neg=5, batch=512, lr=0.025

Nearest neighbours:
```
    king         -> vii(0.965), viii(0.942), crowned(0.940), queen(0.937), alfonso(0.931)
    paris        -> petersburg(0.928), concord(0.923), bologna(0.922), munich(0.920), syracuse(0.919)
    computer     -> graphical(0.935), desktop(0.932), bsd(0.932), graphics(0.932), pdp(0.928)
    good         -> lot(0.932), doing(0.927), everyone(0.920), odds(0.918), feel(0.918)
    one          -> seven(0.905), six(0.889), eight(0.889), four(0.885), five(0.873)
```

---

**Epoch 3/5** — avg loss: `2.3456`

Config: vocab filtered at min_count=5, dim=100, window=2, neg=5, batch=512, lr=0.025

Nearest neighbours:
```
    king         -> vii(0.962), emperor(0.939), viii(0.936), crowned(0.933), constantine(0.928)
    paris        -> bologna(0.916), petersburg(0.912), concord(0.908), munich(0.905), syracuse(0.899)
    computer     -> graphical(0.931), desktop(0.924), design(0.923), hardware(0.920), computers(0.920)
    good         -> lot(0.920), things(0.912), everyone(0.911), worry(0.908), give(0.905)
    one          -> seven(0.892), six(0.879), eight(0.878), four(0.865), five(0.851)
```

<!-- RESULTS_END -->

For a step-by-step explanation of the algorithm with a toy corpus, see [`word2vec.ipynb`](word2vec.ipynb).

---

## References

Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013).
*Distributed representations of words and phrases and their compositionality.*
NeurIPS 2013. https://arxiv.org/abs/1310.4546

Goldberg, Y. & Levy, O. (2014).
*word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method.*
https://arxiv.org/abs/1402.3722

Jurafsky, D. & Martin, J. H. (2024).
*Speech and Language Processing*, 3rd ed. Chapter 6: Vector Semantics and Embeddings.
https://web.stanford.edu/~jurafsky/slp3/

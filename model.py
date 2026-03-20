"""
Skip-Gram with Negative Sampling (SGNS)

Mathematical background
-----------------------
Given a target word w and a context word c, SGNS maximises:

    J_pos = log sigmoid(c · w)

and, for K noise words n_1 ... n_K drawn from p_n(w) ∝ count(w)^0.75:

    J_neg = Σ_{k=1}^{K}  log sigmoid(-n_i · w)

The objective for one (target, context) pair is J = J_pos + J_neg.
We *minimise* L = -J.

Two embedding matrices are maintained:
  W  [V x D]  - target word embeddings
  C  [V x D]  - context and noise word embeddings

Gradients (per sample)
----------------------
∂L/∂w    =  -(1 - sigmoid(c · w)) c  +  Σ_k sigmoid(n_k · w) n_k
∂L/∂c    =  -(1 - sigmoid(c · w)) w
∂L/∂n_k  =      sigmoid(n_k · w)  w

In vectorised form for a batch of size B with K negatives each:

    d_pos  = sigmoid(z_pos) - 1          shape [B]   per-sample
    d_neg  = sigmoid(z_neg)              shape [B, K] per-sample

    grad_w  =  d_pos[:,None] * c  +  (d_neg[:,:,None] * n).sum(1)
    grad_c  =  d_pos[:,None] * w
    grad_n  =  d_neg[:,:,None] * w[:,None,:]

Gradients are per-sample (not batch-averaged). numpy.add.at scatters each
sample's gradient independently, so a batch of B pairs is equivalent to B
consecutive per-sample SGD steps - matching the original word2vec behaviour
where lr is applied once per (target, context, noise) tuple.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Numerically stable sigmoid
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Element-wise sigmoid, stable for large |x|.

    Uses exp(-|x|) which is always in [0, 1], avoiding overflow:
        sigmoid(x) = 1 / (1 + exp(-|x|))        for x >= 0
        sigmoid(x) = exp(-|x|) / (1 + exp(-|x|)) for x < 0
    """
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


# ---------------------------------------------------------------------------
# Word2Vec model
# ---------------------------------------------------------------------------

class Word2Vec:
    """
    Skip-gram word2vec trained with negative sampling (SGNS).

    Parameters
    ----------
    vocab_size : number of words in the vocabulary
    embed_dim  : dimensionality of word embeddings (D)
    seed       : RNG seed for reproducibility
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        seed: int = 42,
        dtype: np.dtype = np.float32,
    ):
        rng = np.random.default_rng(seed)
        self.dtype = np.dtype(dtype)

        # W - target word embeddings; initialised uniformly in
        # [-0.5/D, 0.5/D] following the original word2vec C code.
        self.W: np.ndarray = rng.uniform(
            -0.5 / embed_dim, 0.5 / embed_dim, (vocab_size, embed_dim)
        ).astype(self.dtype)

        # C - context and noise word embeddings; zero-initialised.
        self.C: np.ndarray = np.zeros(
            (vocab_size, embed_dim), dtype=self.dtype
        )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    # ------------------------------------------------------------------
    # Forward pass + loss + gradient computation
    # ------------------------------------------------------------------

    def forward_backward(
        self,
        targets: np.ndarray,    # int32 [B]
        contexts: np.ndarray,   # int32 [B]
        negatives: np.ndarray,  # int32 [B, K]
    ) -> tuple[float, dict]:
        """
        Compute the SGNS loss and gradients for one mini-batch.

        Returns
        -------
        loss   : mean per-sample loss over the batch (for monitoring)
        grads  : dict with keys 'w', 'c', 'n' (per-sample gradient
                 arrays) and 'targets', 'contexts', 'negatives' (index arrays
                 for the scatter-add update step).
        """
        # ---- Look up embeddings ----------------------------------------
        w = self.W[targets]          # [B, D]    target word vectors
        c = self.C[contexts]         # [B, D]    context word vectors
        n = self.C[negatives]        # [B, K, D] noise word vectors

        # ---- Dot products (scores) -------------------------------------
        # z_pos[b] = c[b] · w[b]
        z_pos = (w * c).sum(axis=1)                        # [B]
        # z_neg[b, k] = n[b, k] · w[b]
        z_neg = (n * w[:, np.newaxis, :]).sum(axis=2)      # [B, K]

        # ---- Loss (mean per-sample, for monitoring only) ---------------
        sig_pos = sigmoid(z_pos)          # [B]
        sig_neg = sigmoid(z_neg)          # [B, K]

        loss_pos = -np.log(sig_pos + 1e-10).mean()
        loss_neg = -np.log(1.0 - sig_neg + 1e-10).sum(axis=1).mean()
        loss = float(loss_pos + loss_neg)

        # ---- Per-sample gradients --------------------------------------
        #
        # ∂L/∂z_pos[b]   = sigmoid(z_pos[b]) - 1
        # ∂L/∂z_neg[b,k] = sigmoid(z_neg[b,k])
        #
        # Not divided by B - see module docstring.
        d_pos = sig_pos - 1.0                # [B]
        d_neg = sig_neg                      # [B, K]

        # ∂L/∂w[b]   = d_pos[b]·c[b] + Σ_k d_neg[b,k]·n[b,k]
        grad_w = (
            d_pos[:, np.newaxis] * c                          # pos term
            + (d_neg[:, :, np.newaxis] * n).sum(axis=1)       # neg terms
        )  # [B, D]

        # ∂L/∂c[b] = d_pos[b] · w[b]
        grad_c = d_pos[:, np.newaxis] * w                     # [B, D]

        # ∂L/∂n[b,k] = d_neg[b,k] · w[b]
        grad_n = d_neg[:, :, np.newaxis] * w[:, np.newaxis, :]  # [B, K, D]

        grads = {
            "w": grad_w,
            "c": grad_c,
            "n": grad_n,
            "targets": targets,
            "contexts": contexts,
            "negatives": negatives,
        }
        return loss, grads

    # ------------------------------------------------------------------
    # Parameter update - SGD with scatter-add
    # ------------------------------------------------------------------

    def sgd_step(self, grads: dict, lr: float) -> None:
        """
        Apply one SGD update given the gradient dict from forward_backward.

        numpy.add.at is used instead of fancy-indexing assignment so that
        duplicate indices within a batch contribute additively (i.e. their
        gradients are summed before being applied), which is the correct
        behaviour for stochastic gradient descent on embedding tables.
        """
        targets   = grads["targets"]
        contexts  = grads["contexts"]
        negatives = grads["negatives"]   # [B, K]

        # Update target word embeddings (W)
        np.add.at(self.W, targets,  -lr * grads["w"])

        # Update context word embeddings (C rows for context words)
        np.add.at(self.C, contexts, -lr * grads["c"])

        # Update noise word embeddings (C rows for noise words)
        # Flatten [B, K] -> [B*K] to use a single scatter-add call.
        neg_flat   = negatives.reshape(-1)                       # [B*K]
        grad_n_flat = grads["n"].reshape(-1, self.embed_dim)     # [B*K, D]
        np.add.at(self.C, neg_flat, -lr * grad_n_flat)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_embeddings(self) -> np.ndarray:
        """Return the averaged target+context embeddings (optional fusion)."""
        return 0.5 * (self.W + self.C)

    def most_similar(
        self,
        word: str,
        word2idx: dict,
        idx2word: list,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Return the top-k most similar words by cosine similarity.
        Uses target embeddings W only.
        """
        if word not in word2idx:
            raise KeyError(f"'{word}' not in vocabulary")

        idx = word2idx[word]
        vec = self.W[idx]                     # [D]

        # Cosine similarity = dot product of L2-normalised vectors
        norms = np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-10
        normed = self.W / norms
        scores = normed @ (vec / (np.linalg.norm(vec) + 1e-10))  # [V]
        scores[idx] = -np.inf  # exclude the query word itself

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(idx2word[i], float(scores[i])) for i in top_indices]

"""
Data loading and preprocessing for word2vec training.

Downloads the text8 corpus (first 100 MB of cleaned Wikipedia text),
builds a vocabulary, subsamples frequent words, constructs a negative-
sampling distribution, and yields mini-batches for the training loop.
"""

import os
import urllib.request
import zipfile
from collections import Counter

import numpy as np


DATA_URL = "http://mattmahoney.net/dc/text8.zip"
DATA_DIR = "data"

# Download and preprocess

def download_text8(data_dir: str = DATA_DIR) -> list[str]:
    """Return text8 as a list of tokens"""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "text8.zip")
    txt_path = os.path.join(data_dir, "text8")

    if not os.path.exists(txt_path):
        print(f"Downloading text8 from {DATA_URL} …")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
        print("Download complete.")

    with open(txt_path, "r", encoding="utf-8") as f:
        tokens = f.read().split()

    print(f"Corpus size: {len(tokens):,} tokens")
    return tokens


# Vocabulary

def build_vocab(
    tokens: list[str],
    min_count: int = 5,
) -> tuple[dict[str, int], list[str], np.ndarray]:
    """
    Build word-index mappings and word frequency counts.

    Parameters
    ----------
    tokens     : list of word strings
    min_count  : words appearing fewer times are discarded

    Returns
    -------
    word2idx  : {word: int_index} sorted by descending frequency
    idx2word  : [word_at_index_0, word_at_index_1, …]
    counts    : int64 array of shape [vocab_size]
    """
    raw_counts = Counter(tokens)
    # Sort by frequency
    vocab_items = sorted(
        [(w, c) for w, c in raw_counts.items() if c >= min_count],
        key=lambda x: -x[1],
    )

    word2idx: dict[str, int] = {w: i for i, (w, _) in enumerate(vocab_items)}
    idx2word: list[str] = [w for w, _ in vocab_items]
    counts = np.array([c for _, c in vocab_items], dtype=np.int64)

    print(f"Vocabulary size: {len(word2idx):,}  (min_count={min_count})")
    return word2idx, idx2word, counts


# Subsampling

def subsample_tokens(
    tokens: list[str],
    word2idx: dict[str, int],
    counts: np.ndarray,
    t: float = 1e-5,
) -> np.ndarray:
    """
    Apply Mikolov et al. (2013) frequent-word subsampling.
    Each token w is kept with probability

        P(keep | w) = min(1,  sqrt(t / f(w))  +  t / f(w))

    where f(w) = count(w) / total_tokens.  Rare words are almost always
    kept; very frequent words (e.g. "the", "of") are aggressively thinned.

    Returns an int32 array of word indices (OOV tokens already removed).
    """
    total = counts.sum()
    freq = counts / total
    keep_prob = np.minimum(1.0, np.sqrt(t / freq) + t / freq)

    indices: list[int] = []
    for w in tokens:
        idx = word2idx.get(w)
        if idx is None:
            continue  # OOV - skip
        if np.random.random() < keep_prob[idx]:
            indices.append(idx)

    result = np.array(indices, dtype=np.int32)
    print(
        f"After subsampling: {len(result):,} tokens "
        f"(kept {100 * len(result) / len(tokens):.1f}%)"
    )
    return result


# Negative-sampling distribution

def make_neg_sampler(counts: np.ndarray, power: float = 0.75):
    """
    Return a callable that samples word indices from the unigram^power
    distribution used for negative sampling in SGNS.

    Raising raw counts to the power 0.75 (from Mikolov et al. 2013)
    smooths the distribution: very frequent words are still sampled
    often, but rare words get a larger share than under true unigram.
    """
    weights = counts.astype(np.float64) ** power
    weights = weights / weights.sum()
    vocab_size = len(counts)

    def sample(size: int | tuple) -> np.ndarray:
        return np.random.choice(vocab_size, size=size, p=weights)

    return sample


# Mini-batch generator

def batch_iter(
    tokens: np.ndarray,
    batch_size: int,
    window_size: int,
    neg_sampler,
    n_negatives: int,
):
    """
    Streaming mini-batch generator for skip-gram training.

    For each target word, all tokens within a fixed window of radius
    window_size are used as positive context words.

    Yields
    ------
    targets   : int32 [B]        - target word indices
    contexts  : int32 [B]        - positive context word indices
    negatives : int32 [B, K]     - noise word indices
    """
    n = len(tokens)
    targets_buf: list[int] = []
    contexts_buf: list[int] = []
    negatives_buf: list[np.ndarray] = []

    for i in range(n):
        target = int(tokens[i])
        lo, hi = max(0, i - window_size), min(n, i + window_size + 1)

        for j in range(lo, hi):
            if j == i:
                continue
            targets_buf.append(target)
            contexts_buf.append(int(tokens[j]))
            negatives_buf.append(neg_sampler(n_negatives))

            if len(targets_buf) == batch_size:
                yield (
                    np.array(targets_buf, dtype=np.int32),
                    np.array(contexts_buf, dtype=np.int32),
                    np.stack(negatives_buf).astype(np.int32),
                )
                targets_buf = []
                contexts_buf = []
                negatives_buf = []

    if targets_buf:
        yield (
            np.array(targets_buf, dtype=np.int32),
            np.array(contexts_buf, dtype=np.int32),
            np.stack(negatives_buf).astype(np.int32),
        )

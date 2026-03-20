"""
Training script for skip-gram word2vec with negative sampling.

Usage
-----
    python train.py [--embed_dim D] [--window W] [--n_neg K]
                    [--batch_size B] [--epochs E] [--lr LR]
                    [--min_count M] [--subsample_t T]
                    [--save_path PATH] [--seed S]

All arguments are optional; reasonable defaults are provided.

Training schedule
-----------------
The learning rate follows the linear decay used in the original word2vec
C code:

    lr(step) = lr_0 * max(0.0001,  1 - step / total_steps)

After each epoch a short nearest-neighbour report is printed as a sanity
check. Embeddings are saved to disk (numpy .npz) after the final epoch.
"""

import argparse
import os
import time

import numpy as np

from data import download_text8, build_vocab, subsample_tokens, make_neg_sampler, batch_iter
from model import Word2Vec


# Helpers

def linear_lr(step: int, total_steps: int, lr_0: float) -> float:
    """Linearly decaying learning rate, floored at lr_0 * 1e-4."""
    progress = step / max(total_steps, 1)
    return lr_0 * max(1e-4, 1.0 - progress)


def save(model: Word2Vec, idx2word: list, path: str) -> None:
    np.savez(
        path,
        W=model.W,
        C=model.C,
        idx2word=np.array(idx2word),
    )
    print(f"  Saved embeddings → {path}.npz")


def quick_eval(model: Word2Vec, word2idx: dict, idx2word: list) -> list[str]:
    """Print nearest neighbours for a handful of probe words. Returns result lines."""
    probes = ["king", "paris", "computer", "good", "one"]
    lines = []
    for word in probes:
        if word not in word2idx:
            continue
        neighbours = model.most_similar(word, word2idx, idx2word, top_k=5)
        nn_str = ", ".join(f"{w}({s:.3f})" for w, s in neighbours)
        line = f"    {word:12s} -> {nn_str}"
        print(line)
        lines.append(line.strip())
    return lines


README = os.path.join(os.path.dirname(__file__), "README.md")
_START = "<!-- RESULTS_START -->"
_END   = "<!-- RESULTS_END -->"

def log_results_to_readme(epoch: int, epochs: int, avg_loss: float, nn_lines: list[str], args: argparse.Namespace) -> None:
    """Replace the RESULTS block in README.md with the latest training log."""
    if not os.path.exists(README):
        return
    with open(README, encoding="utf-8") as f:
        text = f.read()
    start = text.find(_START)
    end   = text.find(_END)
    if start == -1 or end == -1:
        return

    config = (
        f"vocab filtered at min_count={args.min_count}, "
        f"dim={args.embed_dim}, window={args.window}, "
        f"neg={args.n_neg}, batch={args.batch_size}, lr={args.lr}"
    )
    nn_block = "\n".join(f"    {l}" for l in nn_lines)
    entry = (
        f"**Epoch {epoch}/{epochs}** — avg loss: `{avg_loss:.4f}`\n\n"
        f"Config: {config}\n\n"
        f"Nearest neighbours:\n```\n{nn_block}\n```"
    )

    # Collect previous entries (everything between the markers, minus any existing entry for this epoch)
    existing = text[start + len(_START):end].strip()
    prev_entries = [e for e in existing.split("\n\n---\n\n") if e and not e.startswith(f"**Epoch {epoch}/")]
    all_entries = prev_entries + [entry]
    block = "\n\n---\n\n".join(all_entries)

    new_text = text[:start + len(_START)] + "\n\n" + block + "\n\n" + text[end:]
    with open(README, "w", encoding="utf-8") as f:
        f.write(new_text)


# Training loop

def train(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)

    # Data
    tokens_raw = download_text8()
    word2idx, idx2word, counts = build_vocab(tokens_raw, min_count=args.min_count)
    tokens = subsample_tokens(tokens_raw, word2idx, counts, t=args.subsample_t)
    neg_sampler = make_neg_sampler(counts, power=0.75)

    # Model
    model = Word2Vec(
        vocab_size=len(word2idx),
        embed_dim=args.embed_dim,
        seed=args.seed,
    )
    print(
        f"\nModel: vocab={len(word2idx):,}  dim={args.embed_dim}"
        f"  window={args.window}  neg={args.n_neg}"
        f"  batch={args.batch_size}  lr={args.lr}\n"
    )

    # Estimate total steps for the LR schedule
    pairs_per_epoch = len(tokens) * args.window  # rough estimate
    total_steps = args.epochs * (pairs_per_epoch // args.batch_size)

    # Training loop
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_batches = 0
        t0 = time.time()

        for targets, contexts, negatives in batch_iter(
            tokens,
            batch_size=args.batch_size,
            window_size=args.window,
            neg_sampler=neg_sampler,
            n_negatives=args.n_neg,
        ):
            lr = linear_lr(global_step, total_steps, args.lr)

            loss, grads = model.forward_backward(targets, contexts, negatives)
            model.sgd_step(grads, lr)

            epoch_loss += loss
            epoch_batches += 1
            global_step += 1

            if global_step % 10_000 == 0:
                avg = epoch_loss / epoch_batches
                elapsed = time.time() - t0
                print(
                    f"  epoch {epoch}  step {global_step:,}"
                    f"  loss={avg:.4f}  lr={lr:.6f}  {elapsed:.0f}s"
                )

        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"\nEpoch {epoch}/{args.epochs}  avg_loss={avg_loss:.4f}\n")
        nn_lines = quick_eval(model, word2idx, idx2word)
        log_results_to_readme(epoch, args.epochs, avg_loss, nn_lines, args)
        if epoch == args.epochs:
            save(model, idx2word, args.save_path)
        print()


# CLI

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train skip-gram word2vec")
    p.add_argument("--embed_dim",    type=int,   default=100)
    p.add_argument("--window",       type=int,   default=2)
    p.add_argument("--n_neg",        type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--lr",           type=float, default=0.025)
    p.add_argument("--min_count",    type=int,   default=5)
    p.add_argument("--subsample_t",  type=float, default=1e-5)
    p.add_argument("--save_path",    type=str,   default="embeddings")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

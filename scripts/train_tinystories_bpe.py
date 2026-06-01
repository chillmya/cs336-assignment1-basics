from __future__ import annotations

import argparse
import json
import pickle
import resource
import threading
import time
from pathlib import Path

from cs336_basics.train_bpe import train_bpe


def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs], strict=True))


def as_gpt2_token(token: bytes, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[b] for b in token)


def start_heartbeat(start: float, done: threading.Event) -> None:
    def heartbeat() -> None:
        while not done.wait(60):
            elapsed_min = (time.time() - start) / 60
            print(f"[train_bpe_artifact] still running: {elapsed_min:.1f} min", flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()


def write_outputs(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: Path,
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str],
    elapsed_s: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "vocab.pkl").open("wb") as f:
        pickle.dump(vocab, f)
    with (output_dir / "merges.pkl").open("wb") as f:
        pickle.dump(merges, f)

    byte_encoder = gpt2_bytes_to_unicode()
    readable_vocab = {as_gpt2_token(token, byte_encoder): token_id for token_id, token in vocab.items()}
    with (output_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(readable_vocab, f, ensure_ascii=False, indent=2)

    with (output_dir / "merges.txt").open("w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{as_gpt2_token(left, byte_encoder)} {as_gpt2_token(right, byte_encoder)}\n")

    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    metadata = {
        "input_path": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "num_vocab_items": len(vocab),
        "num_merges": len(merges),
        "elapsed_seconds": elapsed_s,
        "max_rss_mb": max_rss_mb,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tinystories_bpe"))
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--special-token", action="append", default=None)
    args = parser.parse_args()

    special_tokens = args.special_token or ["<|endoftext|>"]
    start = time.time()
    done = threading.Event()
    start_heartbeat(start, done)

    print(
        "[train_bpe_artifact] starting "
        f"input={args.input} vocab_size={args.vocab_size} output_dir={args.output_dir}",
        flush=True,
    )
    try:
        vocab, merges = train_bpe(args.input, args.vocab_size, special_tokens)
    finally:
        done.set()

    elapsed_s = time.time() - start
    write_outputs(vocab, merges, args.output_dir, args.input, args.vocab_size, special_tokens, elapsed_s)
    print(
        "[train_bpe_artifact] done "
        f"elapsed={elapsed_s / 60:.1f} min vocab={len(vocab)} merges={len(merges)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

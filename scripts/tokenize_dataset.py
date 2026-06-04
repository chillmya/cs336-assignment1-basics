from __future__ import annotations

import argparse
import codecs
import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


SPECIAL_TOKEN = "<|endoftext|>"

DATASETS = {
    "tinystories": {
        "tokenizer_dir": Path("artifacts/tinystories_bpe"),
        "splits": {
            "train": Path("data/TinyStoriesV2-GPT4-train.txt"),
            "valid": Path("data/TinyStoriesV2-GPT4-valid.txt"),
        },
    },
    "owt": {
        "tokenizer_dir": Path("artifacts/owt_bpe"),
        "splits": {
            "train": Path("data/owt_train.txt"),
            "valid": Path("data/owt_valid.txt"),
        },
    },
}


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    return Tokenizer.from_files(
        str(tokenizer_dir / "vocab.pkl"),
        str(tokenizer_dir / "merges.pkl"),
        [SPECIAL_TOKEN],
    )


def iter_token_chunks(
    input_path: Path,
    tokenizer: Tokenizer,
    chunk_bytes: int,
    max_bytes: int | None = None,
) -> Iterator[tuple[list[int], int]]:
    """Yield token-id chunks and the number of input bytes consumed.

    Text is read as bytes and decoded incrementally so UTF-8 characters can
    safely cross read boundaries. We split on the dataset special token and
    tokenize one complete document at a time, preserving exact special-token IDs.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    special_id = tokenizer.special_token_ids[SPECIAL_TOKEN]

    with input_path.open("rb") as f:
        while True:
            to_read = chunk_bytes
            if max_bytes is not None:
                remaining = max_bytes - f.tell()
                if remaining <= 0:
                    break
                to_read = min(to_read, remaining)

            raw = f.read(to_read)
            if not raw:
                break

            text = decoder.decode(raw)
            buffer += text
            consumed = len(raw)

            while SPECIAL_TOKEN in buffer:
                document, _, buffer = buffer.partition(SPECIAL_TOKEN)
                ids = tokenizer.encode(document)
                ids.append(special_id)
                yield ids, consumed
                consumed = 0

            if consumed:
                yield [], consumed

    tail = decoder.decode(b"", final=True)
    buffer += tail
    if buffer:
        yield tokenizer.encode(buffer), 0


def benchmark(input_path: Path, tokenizer: Tokenizer, max_bytes: int, chunk_bytes: int) -> dict[str, float | int]:
    start = time.time()
    total_bytes = 0
    total_tokens = 0

    for ids, consumed in iter_token_chunks(input_path, tokenizer, chunk_bytes, max_bytes):
        total_bytes += consumed
        total_tokens += len(ids)

    elapsed = time.time() - start
    return {
        "bytes": total_bytes,
        "tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "bytes_per_second": total_bytes / elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "bytes_per_token": total_bytes / total_tokens,
    }


def write_bin(
    input_path: Path,
    output_path: Path,
    tokenizer: Tokenizer,
    chunk_bytes: int,
    flush_tokens: int,
) -> dict[str, float | int | str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    total_bytes = 0
    total_tokens = 0
    pending: list[int] = []

    with output_path.open("wb") as out:
        for ids, consumed in iter_token_chunks(input_path, tokenizer, chunk_bytes):
            total_bytes += consumed
            pending.extend(ids)

            if len(pending) >= flush_tokens:
                np.asarray(pending, dtype=np.uint16).tofile(out)
                total_tokens += len(pending)
                pending.clear()

        if pending:
            np.asarray(pending, dtype=np.uint16).tofile(out)
            total_tokens += len(pending)

    elapsed = time.time() - start
    metadata = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "dtype": "uint16",
        "special_token": SPECIAL_TOKEN,
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
        "elapsed_seconds": elapsed,
        "bytes_per_second": total_bytes / elapsed,
        "tokens_per_second": total_tokens / elapsed,
    }

    with output_path.with_suffix(output_path.suffix + ".json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--split", choices=["train", "valid"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/tokenized"))
    parser.add_argument("--chunk-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--flush-tokens", type=int, default=1_000_000)
    parser.add_argument("--benchmark-bytes", type=int, default=None)
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    input_path = config["splits"][args.split]
    tokenizer = load_tokenizer(config["tokenizer_dir"])

    if args.benchmark_bytes is not None:
        result = benchmark(input_path, tokenizer, args.benchmark_bytes, args.chunk_bytes)
        print(json.dumps(result, indent=2))
        return

    output_path = args.output_dir / f"{args.dataset}_{args.split}.bin"
    metadata = write_bin(input_path, output_path, tokenizer, args.chunk_bytes, args.flush_tokens)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

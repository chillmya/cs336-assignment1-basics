from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import regex as re


GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# A pre-token is stored as a tuple of current BPE token byte strings.
# Example: " text" starts as (b" ", b"t", b"e", b"x", b"t").
Pretoken = tuple[bytes, ...]
Pair = tuple[bytes, bytes]


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    This is intentionally a skeleton. Fill in the TODOs in the helper functions
    first, then wire up the merge loop here.
    """
    # Start with the 256 possible byte values, plus any special tokens.
    # Example: b"a" is already present at byte id 97.
    vocab = _initial_vocab(special_tokens)

    # Keep the sequence of merges in training order.
    # Example: [(b"t", b"h"), (b"th", b"e")].
    merges: list[Pair] = []

    # Count how many times each GPT-2-style pre-token appears in the corpus.
    # Example: "hello hello" may contribute the same pretoken twice.
    pretoken_counts = _count_pretokens(input_path, special_tokens)
    pair_counts, pair_to_pretokens = _count_pairs_and_pretokens(pretoken_counts)

    # Each merge adds exactly one vocabulary item. Stop when the requested final
    # vocabulary size is reached, including special tokens.
    while len(vocab) < vocab_size and pair_counts:
        # Choose the most frequent pair; ties prefer the lexicographically greater pair.
        # Example: if (b"a", b"b") and (b"a", b"c") both appear 10 times, choose (b"a", b"c").
        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]

        # Save the selected merge rule.
        # Example: (b"t", b"h") means merge b"t" followed by b"h".
        merges.append(best_pair)

        # Build the new merged token bytes.
        # Example: b"t" + b"h" -> b"th".
        merged_token = best_pair[0] + best_pair[1]

        # Insert the merged token into the vocabulary at the next available id.
        # Example: if len(vocab) is 257, the new token goes at key 257.
        vocab[len(vocab)] = merged_token

        # Recompute pretokens after replacing every occurrence of the chosen pair.
        # Example: (b"t", b"h", b"e") becomes (b"th", b"e").
        pretoken_counts = _merge_pair_in_pretokens(
            pretoken_counts,
            best_pair,
            pair_to_pretokens[best_pair],
            pair_counts,
            pair_to_pretokens,
        )

    return vocab, merges


def _initial_vocab(special_tokens: Iterable[str]) -> dict[int, bytes]:
    """Create the initial byte vocabulary plus any special tokens."""
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    return vocab


def _count_pretokens(input_path: str | os.PathLike, special_tokens: list[str]) -> Counter[Pretoken]:
    """Read the corpus and count pre-token byte sequences.

    Special tokens should act as hard boundaries and should not themselves
    contribute to BPE merge statistics.
    """
    pretoken_counts: Counter[Pretoken] = Counter()

    if len(special_tokens) == 1:
        special_token_bytes = special_tokens[0].encode("utf-8")
        buffer = b""
        with Path(input_path).open("rb") as f:
            while chunk := f.read(1024 * 1024):
                pieces = (buffer + chunk).split(special_token_bytes)
                for segment_bytes in pieces[:-1]:
                    _count_pretokens_in_segment(segment_bytes.decode("utf-8"), pretoken_counts)
                buffer = pieces[-1]

        if buffer:
            _count_pretokens_in_segment(buffer.decode("utf-8"), pretoken_counts)
    else:
        text = Path(input_path).read_text(encoding="utf-8")
        for segment in _split_on_special_tokens(text, special_tokens):
            _count_pretokens_in_segment(segment, pretoken_counts)

    return pretoken_counts


def _count_pretokens_in_segment(segment: str, pretoken_counts: Counter[Pretoken]) -> None:
    """Count GPT-2-style pre-tokens from a text segment with no special tokens."""
    for match in re.finditer(GPT2_PRETOKEN_PATTERN, segment):
        token_bytes = match.group(0).encode("utf-8")
        pretoken = tuple(bytes([b]) for b in token_bytes)
        pretoken_counts[pretoken] += 1


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Split text so BPE never merges across a special token boundary."""
    if not special_tokens:
        return [text]

    escaped_tokens = [re.escape(token) for token in special_tokens]
    delimiter = "|".join(escaped_tokens)
    return re.split(delimiter, text)


def _count_pairs(pretoken_counts: Counter[Pretoken]) -> Counter[Pair]:
    """Count adjacent token pairs, weighted by pre-token frequency."""
    pair_counts, _ = _count_pairs_and_pretokens(pretoken_counts)
    return pair_counts


def _count_pairs_and_pretokens(
    pretoken_counts: Counter[Pretoken],
) -> tuple[Counter[Pair], dict[Pair, set[Pretoken]]]:
    """Count adjacent token pairs and remember which pre-tokens contain them."""
    pair_counts: Counter[Pair] = Counter()
    pair_to_pretokens: dict[Pair, set[Pretoken]] = {}

    for pretoken, count in pretoken_counts.items():
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_counts[pair] += count
            pair_to_pretokens.setdefault(pair, set()).add(pretoken)

    return pair_counts, pair_to_pretokens


def _merge_pair_in_pretokens(
    pretoken_counts: Counter[Pretoken],
    pair_to_merge: Pair,
    affected_pretokens: set[Pretoken] | None = None,
    pair_counts: Counter[Pair] | None = None,
    pair_to_pretokens: dict[Pair, set[Pretoken]] | None = None,
) -> Counter[Pretoken]:
    """Apply one BPE merge to every pre-token sequence."""
    if affected_pretokens is None:
        affected_pretokens = set(pretoken_counts)

    if pair_counts is not None and pair_to_pretokens is not None:
        return _merge_pair_in_pretokens_incremental(
            pretoken_counts,
            pair_to_merge,
            affected_pretokens,
            pair_counts,
            pair_to_pretokens,
        )

    merged_counts = pretoken_counts.copy()
    left, right = pair_to_merge
    merged_token = left + right

    for pretoken in affected_pretokens:
        count = merged_counts.pop(pretoken)
        merged_pretoken: list[bytes] = []
        i = 0
        while i < len(pretoken):
            if i + 1 < len(pretoken) and pretoken[i] == left and pretoken[i + 1] == right:
                merged_pretoken.append(merged_token)
                i += 2
            else:
                merged_pretoken.append(pretoken[i])
                i += 1

        merged_counts[tuple(merged_pretoken)] += count

    return merged_counts


def _merge_pair_in_pretokens_incremental(
    pretoken_counts: Counter[Pretoken],
    pair_to_merge: Pair,
    affected_pretokens: set[Pretoken],
    pair_counts: Counter[Pair],
    pair_to_pretokens: dict[Pair, set[Pretoken]],
) -> Counter[Pretoken]:
    """Apply one merge and update cached pair statistics for changed pre-tokens."""
    old_counts = {pretoken: pretoken_counts[pretoken] for pretoken in affected_pretokens}
    new_counts: Counter[Pretoken] = Counter()

    for pretoken, count in old_counts.items():
        del pretoken_counts[pretoken]
        _remove_pretoken_pair_counts(pretoken, count, pair_counts, pair_to_pretokens)
        new_counts[_merge_pretoken(pretoken, pair_to_merge)] += count

    for pretoken, count in new_counts.items():
        pretoken_counts[pretoken] += count
        _add_pretoken_pair_counts(pretoken, count, pair_counts, pair_to_pretokens)

    return pretoken_counts


def _merge_pretoken(pretoken: Pretoken, pair_to_merge: Pair) -> Pretoken:
    """Return one pre-token with every non-overlapping occurrence of a pair merged."""
    left, right = pair_to_merge
    merged_token = left + right
    merged_pretoken: list[bytes] = []
    i = 0
    while i < len(pretoken):
        if i + 1 < len(pretoken) and pretoken[i] == left and pretoken[i + 1] == right:
            merged_pretoken.append(merged_token)
            i += 2
        else:
            merged_pretoken.append(pretoken[i])
            i += 1

    return tuple(merged_pretoken)


def _add_pretoken_pair_counts(
    pretoken: Pretoken,
    count: int,
    pair_counts: Counter[Pair],
    pair_to_pretokens: dict[Pair, set[Pretoken]],
) -> None:
    """Add one pre-token's weighted pair counts to the cached statistics."""
    for i in range(len(pretoken) - 1):
        pair = (pretoken[i], pretoken[i + 1])
        pair_counts[pair] += count
        pair_to_pretokens.setdefault(pair, set()).add(pretoken)


def _remove_pretoken_pair_counts(
    pretoken: Pretoken,
    count: int,
    pair_counts: Counter[Pair],
    pair_to_pretokens: dict[Pair, set[Pretoken]],
) -> None:
    """Remove one pre-token's weighted pair counts from the cached statistics."""
    for i in range(len(pretoken) - 1):
        pair = (pretoken[i], pretoken[i + 1])
        new_count = pair_counts[pair] - count
        if new_count:
            pair_counts[pair] = new_count
        else:
            del pair_counts[pair]

        pretokens = pair_to_pretokens.get(pair)
        if pretokens is not None:
            pretokens.discard(pretoken)
            if not pretokens:
                del pair_to_pretokens[pair]

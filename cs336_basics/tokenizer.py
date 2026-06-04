from __future__ import annotations

import pickle
from collections.abc import Iterable, Iterator

import regex as re


GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

Pair = tuple[bytes, bytes]


class Tokenizer:
    """Byte-level BPE tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.token_to_id: dict[bytes, int] = {token: idx for idx, token in vocab.items()}

        self.merge_ranks: dict[Pair, int] = {pair: rank for rank, pair in enumerate(merges)}

        # Sort longest first so overlapping special tokens prefer the longest match.
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        self.special_token_ids: dict[str, int] = {
            token: self.token_to_id[token.encode("utf-8")] for token in self.special_tokens
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Construct a tokenizer from pickle-serialized vocab and merges files."""
        with open(vocab_filepath, "rb") as f:
            vocab: dict[int, bytes] = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges: list[Pair] = pickle.load(f)

        existing_tokens = set(vocab.values())
        for special_token in special_tokens or []:
            encoded = special_token.encode("utf-8")
            if encoded not in existing_tokens:
                vocab[len(vocab)] = encoded
                existing_tokens.add(encoded)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode a string into token ids."""
        ids: list[int] = []

        for chunk, is_special in self._split_text_on_special_tokens(text):
            if is_special:
                ids.append(self.special_token_ids[chunk])
                continue

            for match in re.finditer(GPT2_PRETOKEN_PATTERN, chunk):
                token_bytes = match.group(0).encode("utf-8")
                pieces = tuple(bytes([b]) for b in token_bytes)
                pieces = self._apply_bpe_merges(pieces)
                ids.extend(self.token_to_id[piece] for piece in pieces)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of text chunks without materializing all text."""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to a string."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def _split_text_on_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """Return (chunk, is_special_token) pairs."""
        if not self.special_tokens:
            return [(text, False)] if text else []

        result: list[tuple[str, bool]] = []
        i = 0
        while i < len(text):
            matched = None
            for token in self.special_tokens:
                if text.startswith(token, i):
                    matched = token
                    break
            if matched is not None:
                result.append((matched, True))
                i += len(matched)
                continue

            start = i
            while i < len(text):
                if any(text.startswith(token, i) for token in self.special_tokens):
                    break
                i += 1
            if start < i:
                result.append((text[start:i], False))

        return result

    def _apply_bpe_merges(self, pieces: tuple[bytes, ...]) -> tuple[bytes, ...]:
        """Apply learned BPE merges to one pre-token."""
        while True:
            best_pair: Pair | None = None
            best_rank: int | None = None
            for a, b in zip(pieces, pieces[1:]):
                rank = self.merge_ranks.get((a, b))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = (a, b)
            if best_pair is None:
                return pieces
            pieces = self._merge_pair_once(pieces, best_pair)

    @staticmethod
    def _merge_pair_once(pieces: tuple[bytes, ...], pair: Pair) -> tuple[bytes, ...]:
        """Merge every non-overlapping occurrence of pair in pieces."""
        merged: list[bytes] = []
        i = 0
        while i < len(pieces):
            if i + 1 < len(pieces) and pieces[i] == pair[0] and pieces[i + 1] == pair[1]:
                merged.append(pieces[i] + pieces[i + 1])
                i += 2
            else:
                merged.append(pieces[i])
                i += 1
        return tuple(merged)

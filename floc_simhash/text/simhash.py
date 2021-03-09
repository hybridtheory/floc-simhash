import hashlib
from typing import List, Callable


class SimHash:
    def __init__(
        self,
        n_bits: int,
        tokenizer: Callable[[str], List[str]] = lambda x: x.split(" "),
    ):
        """Return a class that performs SimHash.

        The algorithm will compute hashes of `n_bits` bits, splitting documents using `tokenizer`.

        :param n_bits: Number of bits of the returned hashes. This is currently limited to a maximum
        of 128 bits.
        :type n_bits: int
        :param tokenizer: rule to split a given document into a list of tokens, defaults to `lambda
        x: x.split(" ")`.
        :type tokenizer: Callable[[str], List[str]], optional
        :raises ValueError: if `n_bits` is greater than 128.
        """
        if n_bits > 128:
            raise ValueError("Only hashes up to 128 bits are currently supported")

        self.n_bits = n_bits
        self.tokenizer = tokenizer

    def _bitwise_compare(self, hashes: List[int], result: int, bit: int) -> int:

        if bit >= self.n_bits:
            return result

        hash_bits = [h & 1 for h in hashes]
        next_bit = int(sum(hash_bits) / len(hashes) > 0.5)

        return self._bitwise_compare(
            [h >> 1 for h in hashes], result + next_bit * 2 ** bit, bit + 1
        )

    def inthash(self, document: str) -> int:
        """Compute the hash of `document` and return it as an integer.

        :type document: str
        :rtype: int
        """
        tokens: List[bytes] = [w.encode() for w in self.tokenizer(document)]
        md5_hashes: List = [hashlib.md5(token) for token in tokens]
        token_clipped_hashes: List[int] = [
            int(h.hexdigest(), 16) >> (h.digest_size * 8 - self.n_bits)
            for h in md5_hashes
        ]

        return self._bitwise_compare(token_clipped_hashes, 0, 0)

    def hash(self, document: str) -> str:
        """Compute the hash of `document` as a hexadecimal string.

        :type document: str
        :rtype: str
        """
        return hex(self.inthash(document))[2:]

from typing import Callable, List

from .simhash import SimHash


class SortingSimHash(SimHash):
    def __init__(
        self,
        bits_to_keep: int,
        simhash_bits: int,
        tokenizer: Callable[[str], List[str]] = lambda x: x.split(" "),
    ):
        """Return a class that performs SortingLSH.

        First, a SimHash with `simhash_bits` is computed. Then, the first `bits_to_keep` bits are
        kept.

        :type bits_to_keep: int
        :type simhash_bits: int
        :param tokenizer: method to split a given document into a list of tokens, defaults to
        `lambda x: x.split(" ")`
        :type tokenizer: Callable[[str], List[str]], optional
        :raises ValueError: if `bits_to_keep > simhash_bits`.
        """
        if bits_to_keep > simhash_bits:
            raise ValueError("Cannot keep more bits than those given by Simhash")

        self.bits_to_keep = bits_to_keep
        super().__init__(simhash_bits, tokenizer)

    def inthash(self, document: str) -> int:
        """Compute the hash of `document` and return it as an integer.

        :type document: str
        :rtype: int
        """
        simhash = super().inthash(document)
        return simhash >> (self.n_bits - self.bits_to_keep)

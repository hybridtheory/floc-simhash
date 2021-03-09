from typing import Tuple

import pytest

from hypothesis import given
import hypothesis.strategies as st

from floc_simhash.text.sorting import SortingSimHash
from floc_simhash.text.simhash import SimHash


@st.composite
def bits_greater_than_given(draw, max_bits: int) -> Tuple[int, int]:
    n_bits = draw(st.integers(min_value=0, max_value=max_bits))
    bits = draw(st.integers(min_value=n_bits + 1))
    return bits, n_bits


@st.composite
def bits_smaller_than_given(draw, max_bits: int) -> Tuple[int, int]:
    n_bits = draw(st.integers(min_value=0, max_value=max_bits))
    bits = draw(st.integers(min_value=0, max_value=n_bits))
    return bits, n_bits


@st.composite
def bits_fitting_in_bytes(draw, max_value: int) -> Tuple[int, int]:
    n_bytes = draw(st.integers(min_value=1, max_value=max_value))
    bits = draw(st.integers(min_value=1, max_value=n_bytes))
    return bits * 8 * 2, n_bytes


@given(bits_greater_than_given(max_bits=128))
def test_bits_to_keep_is_lower_than_simhash_bits(bits_nbits):
    with pytest.raises(ValueError):
        SortingSimHash(*bits_nbits)


@given(bits_smaller_than_given(max_bits=128), st.text())
def test_sortinglsh_clips_integer_hashes(bits_nbits, document):
    bits, n_bits = bits_nbits
    shift = n_bits - bits
    sorting_hasher = SortingSimHash(bits, n_bits)
    sim_hasher = SimHash(n_bits)

    sort_hash = sorting_hasher.inthash(document)
    sim_hash = sim_hasher.inthash(document)

    assert sim_hash >> shift == sort_hash


@given(bits_smaller_than_given(max_bits=128), st.text())
def test_sortinglsh_returns_expected_bits(bits_nbits, document):
    bits, n_bits = bits_nbits
    hasher = SortingSimHash(bits, n_bits)

    assert 0 <= hasher.inthash(document) < 2 ** bits


@given(st.integers(min_value=1, max_value=128), st.text())
def test_keep_a_single_bit(n_bits, document):
    hasher = SortingSimHash(1, n_bits)
    assert 0 <= hasher.inthash(document) < 2


@given(bits_fitting_in_bytes(max_value=8), st.text())
def test_document_hex_hashing(bits_nbytes, document):
    bits, n_bytes = bits_nbytes
    sortinghash = SortingSimHash(bits, n_bytes * 8 * 2).hash(document)
    simhash = SimHash(n_bytes * 8 * 2).hash(document)

    assert simhash.startswith(sortinghash)

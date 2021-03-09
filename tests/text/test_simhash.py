from hypothesis import given
import hypothesis.strategies as st
import pytest

from floc_simhash.text.simhash import SimHash


@st.composite
def lists_of_ints_of_bounded_bits(draw, max_value: int):
    n_bits = draw(st.integers(min_value=0, max_value=max_value))
    hashes = draw(st.lists(st.integers(min_value=0, max_value=2 ** n_bits), min_size=1))
    return n_bits, hashes


@given(st.integers(min_value=129))
def test_large_number_of_bits_raises(n_bits):
    with pytest.raises(ValueError):
        SimHash(n_bits)


@given(st.integers(min_value=0, max_value=128))
def test_number_of_bits_in_constructor(n_bits):
    assert SimHash(n_bits=n_bits).n_bits == n_bits


@given(st.lists(st.integers(min_value=0, max_value=1), min_size=100, max_size=100))
def test_bitwise_compare_computes_last_bit_accurately(hashes):
    expected_next_bit = int(sum(hashes) / 100 > 0.5)
    hasher = SimHash(n_bits=1)
    assert hasher._bitwise_compare(hashes, 0, 0) == expected_next_bit


@given(lists_of_ints_of_bounded_bits(max_value=240))
def test_bitwise_compare_arbitrary_bytes(bits_hashes):
    n_bits, hashes = bits_hashes
    hasher = SimHash(1)
    hasher.n_bits = n_bits
    simhash = hasher._bitwise_compare(hashes, 0, 0)
    assert 0 <= simhash < 2 ** n_bits


@given(st.integers(min_value=0, max_value=128), st.text())
def test_document_int_hashing(n_bits, document):
    simhash = SimHash(n_bits).inthash(document)
    assert 0 <= simhash < 2 ** n_bits


@given(st.integers(min_value=1, max_value=128), st.text())
def test_document_hex_hashing(n_bits, document):
    simhash = SimHash(n_bits).hash(document)
    assert 0 < len(simhash) <= 2 * (n_bits // 8 + 1)

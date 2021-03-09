import numpy as np

from hypothesis import given
import hypothesis.strategies as st
from hypothesis.extra import numpy as st_np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

from floc_simhash.transformer.geometric import SimHashTransformer

bitsizes = st.integers(min_value=1, max_value=63)
dims = st.integers(min_value=1, max_value=100)
large_dims = dims.filter(lambda x: x >= 20)
dtypes = st.sampled_from([np.int64, np.float64])


@given(
    bitsizes,
    st_np.arrays(dtypes, shape=st.tuples(dims, dims)),
)
def test_random_vectors_shape(n_bits, X):

    vectors = SimHashTransformer(n_bits).fit(X)._vectors
    assert vectors.shape == (X.shape[1], n_bits), "Unexpected unit vectors shape"


@given(
    bitsizes,
    st_np.arrays(dtypes, shape=st.tuples(dims, large_dims)),
)
def test_random_vectors_are_unit(n_bits, X):
    vectors = SimHashTransformer(n_bits).fit(X)._vectors

    norms = np.linalg.norm(vectors, axis=0)
    assert norms.shape[0] == n_bits, "There should be n_bits unit vectors picked"
    assert all(np.abs(norms - 1) <= 0.0001), "Random vectors should be unitary"


@given(
    bitsizes,
    st_np.arrays(
        dtypes,
        shape=st.tuples(large_dims, large_dims),
    ),
)
def test_hex_strings(n_bits, X):
    cohorts = list(SimHashTransformer(n_bits).fit_transform(X))

    assert all(h[0:2] == "0x" for h in cohorts), "Cohorts expected as hex strings"
    values = [0 <= int(h, 16) < 2 ** n_bits for h in cohorts]
    assert all(values), "Hex strings longer than expected"


@given(
    st.sampled_from([8, 16]),
    st.lists(
        st.lists(
            st.text(
                alphabet=st.characters(blacklist_characters=["|"]),
                max_size=10,
            ),
            min_size=1,
            max_size=100,
        ),
        min_size=1,
        max_size=100,
    ),
)
def test_in_pipeline(n_bits, texts):
    documents = ["|".join(d) for d in texts]
    pipeline = Pipeline(
        [
            (
                "vect",
                CountVectorizer(tokenizer=lambda x: x.split("|"), binary=True),
            ),
            ("simhash", SimHashTransformer(n_bits)),
        ]
    )

    cohorts = pipeline.fit_transform(documents)
    for cohort in cohorts:
        assert cohort[0:2] == "0x"
        assert 0 <= int(cohort, 16) < 2 ** n_bits
